import json
# import qianfan
# import markdown
import os
import time
import random
import copy
import re
import base64
from datetime import datetime
from pathlib import Path
from openai import OpenAI, RateLimitError
# from gradio_client import Client, handle_file
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS  # 允许跨源请求

app = Flask(__name__)
CORS(app)

text_dir = "knowledge_base"
image_dir = "image_base"
prompt_dir = "prompt_template"


def read_config():
    global CONFIG, PROMPT
    # 读取配置
    with open("configNEW.json", "r", encoding="utf-8") as f:
        json_file = json.load(f)
        CONFIG = json_file.get('config', {})
        PROMPT = json_file.get('Prompt_APIs', {})
        # print(CONFIG)
        # print(PROMPT)


read_config()


@app.route('/chat', methods=['POST'])
def chat():
    read_config()
    data = request.get_json()
    user_id = data.get('config', {}).get('id')
    api_type = data.get('config', {}).get('api', '百度')
    chat_Mode = data.get('config', {}).get('chatMode')
    chat_history = data.get('chat', [])

    # 找到最后一条用户消息的 index 和内容
    last_user_index = next((i for i in reversed(range(len(chat_history))) if chat_history[i]["role"] == "user"), None)
    last_user_input = chat_history[last_user_index]["content"] if last_user_index is not None else ""

    # 创建新的 history（不包含最后一条用户输入）
    chat_history = chat_history[:last_user_index] + chat_history[
                                                    last_user_index + 1:] if last_user_index is not None else chat_history.copy()

    conf_list = CONFIG.get(api_type, [])
    if not isinstance(conf_list, list):
        conf_list = [conf_list]

    """def try_process_with_keys(api_type, handler_func, chat_Mode):


        for conf in conf_list:
            try:
                generator = handler_func(user_id, last_user_input, chat_history.copy(), conf, chat_Mode)
                # 如果第一次 chunk 成功，说明当前 key 可用，返回 generator
                first_chunk = next(generator)
                def combined():
                    yield first_chunk
                    for chunk in generator:
                        yield chunk
                return combined()
            except Exception as e:
                print(f"⚠️ {api_type} 尝试失败，切换下一个 key：{str(e)}")
                continue
        return iter([f"<p><b>错误：</b>{api_type} 所有 key 都失败，请检查配置或网络。</p>"])"""

    if api_type == '百度':
        return Response(stream_with_context(
            process_user_input_baidu(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    elif api_type == 'KIMI':
        return Response(stream_with_context(
            process_user_input_KIMI(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    elif api_type == '阿里':
        return Response(stream_with_context(
            process_user_input_ali(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    elif api_type == '腾讯':
        return Response(stream_with_context(
            process_user_input_tengxun(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    elif api_type == 'DeepSeek':
        return Response(stream_with_context(
            process_user_input_deepseek(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    elif api_type == '荔枝':
        return Response(stream_with_context(
            process_user_input_lizhi(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')
    else:
        return Response(stream_with_context(
            process_user_input_baidu(user_id, last_user_input, chat_history.copy(), conf_list, chat_Mode)
        ), content_type='text/plain')


def stream_with_interval(chunk_generator, interval_sec):
    buffer = ""
    last_send_time = 0

    for chunk in chunk_generator:
        buffer += chunk
        current_time = time.time()

        if interval_sec == 0 or current_time - last_send_time >= interval_sec:
            yield buffer
            buffer = ""
            last_send_time = current_time

    if buffer:  # 最后如果还有缓存未发送
        yield buffer


def load_external_content(content: str, base_dir=text_dir) -> str:
    """
    处理content中的文件引用，支持多个文件引用和混合内容
    格式示例："前置内容{file:文件1.txt}中间内容{file:文件2.txt}后缀内容"
    """
    # 匹配所有{file:filename}模式
    file_refs = re.findall(r'\{file:(.*?)\}', content)

    # 逐个加载文件内容
    loaded_contents = {}
    for file_name in file_refs:
        if file_name not in loaded_contents:
            try:
                file_path = Path(base_dir) / file_name
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_contents[file_name] = f.read()
            except Exception as e:
                print(f"加载文件 {file_name} 失败: {e}")
                loaded_contents[file_name] = f"[加载文件 {file_name} 失败]"

    # 替换所有文件引用
    def replace_match(match):
        file_name = match.group(1)
        return loaded_contents.get(file_name, f"[文件 {file_name} 内容缺失]")

    return re.sub(r'\{file:(.*?)\}', replace_match, content)


def prepare_messages(prompt_config, history=None, base_dir=text_dir):
    """准备完整的消息历史，处理所有内容中的文件引用"""
    if history is None:
        history = []

    processed_messages = []
    for message in prompt_config:
        # 深度拷贝消息避免修改原始配置
        processed_msg = message.copy()
        processed_msg["content"] = load_external_content(message["content"], base_dir)
        processed_messages.append(processed_msg)

    return processed_messages + history


def build_full_history_with_references(history, thinking_content, user_input, selected_api, text_dir="knowledge_base",
                                       image_dir="image_base"):
    """
    构建 full_history，附加引用的文本与图片内容，并追加说明性文字
    """
    text_matches = re.findall(r'\{search_text:(.+?)\}', thinking_content)
    image_matches = re.findall(r'\{search_image:(.+?)\}', thinking_content)

    content_blocks = []

    def find_real_filename(search_name, folder_path):
        """
        根据传入的搜索名（可能已修改过），在指定文件夹中还原真实文件名
        """
        # 拆分扩展名（如果有）
        if "." in search_name:
            name_part, ext = os.path.splitext(search_name)
        else:
            name_part, ext = search_name, ""  # 无扩展名

        matched_files = os.listdir(folder_path)
        for real_file in matched_files:
            real_name, real_ext = os.path.splitext(real_file)

            # 扩展名需一致（无扩展名的情况跳过扩展匹配）
            if ext and real_ext.lower() != ext.lower():
                continue
            if not ext and real_ext:  # 搜索名无扩展名，但真实文件有
                continue

            # 判断括号匹配
            bracket_match = re.match(r"^(.*?)\((.*?)\)$", real_name)
            if bracket_match:
                base_name = bracket_match.group(1)
                if base_name == name_part:
                    return real_file
            else:
                if real_name == name_part:
                    return real_file
        return None

    # 处理文本文件
    text_content_list = []
    for search_name in text_matches:
        real_file = find_real_filename(search_name, text_dir)
        if not real_file:
            print(f"⚠ 未找到匹配的文本文件: {search_name}")
            continue

        file_path = os.path.join(text_dir, real_file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                text_content_list.append(f"你可能需要参考的文件：{real_file}\n\n{content}")
            except Exception as e:
                print(f"❌ 读取文本文件失败: {real_file}, 错误: {e}")

    if text_content_list:
        content_blocks.append({
            "type": "text",
            "text": "\n\n".join(text_content_list)
        })

    # 处理图片文件
    for search_name in image_matches:
        real_file = find_real_filename(search_name, image_dir)
        if not real_file:
            print(f"⚠ 未找到匹配的图片文件: {search_name}")
            continue

        file_path = os.path.join(image_dir, real_file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    image_data = f.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")
                mime = real_file.split(".")[-1].lower()
                if mime == "jpg":
                    mime = "jpeg"
                content_blocks.append({
                    "type": "text",
                    "text": f"你可能需要参考的文件：{real_file}"
                })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime};base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"❌ 读取图片文件失败: {real_file}, 错误: {e}")

    # 最后的总结提示
    content_blocks.append({
        "type": "text",
        "text": f"\n\n\n##现在根据上述的要求和你的思维过程正式回复用户，**这次是正式面向用户的回复内容，回答中不要包含思维过程**\n“{user_input}”"
    })

    if selected_api == "baidu":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })
    elif selected_api == "KIMI":
        # 添加这个系统消息
        history.append({
            "role": "system",
            "content": content_blocks
        })
    elif selected_api == "ali":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })
    elif selected_api == "tengxun":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })
    elif selected_api == "deepseek":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })
    elif selected_api == "lizhi":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })

    return history


def build_full_history_with_references_str(history, thinking_content, user_input, selected_api,
                                           text_dir="knowledge_base", image_dir="image_base"):
    """
    构建纯文本版本的 full_history，忽略图片引用，仅合并文本内容
    """
    text_matches = re.findall(r'\{search_text:(.+?)\}', thinking_content)

    content_parts = []

    def find_real_filename(search_name, folder_path):
        """
        根据传入的搜索名（可能已修改过），在指定文件夹中还原真实文件名
        """
        # 拆分扩展名（如果有）
        if "." in search_name:
            name_part, ext = os.path.splitext(search_name)
        else:
            name_part, ext = search_name, ""  # 无扩展名

        matched_files = os.listdir(folder_path)
        for real_file in matched_files:
            real_name, real_ext = os.path.splitext(real_file)

            # 扩展名需一致（无扩展名的情况跳过扩展匹配）
            if ext and real_ext.lower() != ext.lower():
                continue
            if not ext and real_ext:  # 搜索名无扩展名，但真实文件有
                continue

            # 判断括号匹配
            bracket_match = re.match(r"^(.*?)\((.*?)\)$", real_name)
            if bracket_match:
                base_name = bracket_match.group(1)
                if base_name == name_part:
                    return real_file
            else:
                if real_name == name_part:
                    return real_file
        return None

    # 处理文本文件内容
    for search_name in text_matches:
        real_file = find_real_filename(search_name, text_dir)
        if not real_file:
            print(f"⚠ 未找到匹配的文本文件: {search_name}")
            continue

        file_path = os.path.join(text_dir, real_file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content_parts.append(f"你可能需要参考的文件：{real_file}\n\n{content}")
            except Exception as e:
                print(f"❌ 读取文本文件失败: {real_file}, 错误: {e}")

    # 添加总结提示
    content_parts.append(
        f"\n\n\n##现在根据上述的要求和你的思维过程正式回复用户，**这次是正式面向用户的回复内容，回答中不要包含思维过程**\n“{user_input}”")

    # 合并为一个字符串
    full_text = "\n\n\n".join(content_parts)

    if selected_api == "baidu":
        # 添加这个系统消息
        history.append({
            "role": "system",
            "content": full_text
        })
    elif selected_api == "KIMI":
        # 添加这个系统消息
        history.append({
            "role": "system",
            "content": full_text
        })
    elif selected_api == "ali":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": full_text
        })
    elif selected_api == "tengxun":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": full_text
        })
    elif selected_api == "deepseek":
        # 添加这个系统消息
        history.append({
            "role": "system",
            "content": full_text
        })
    elif selected_api == "lizhi":
        # 添加这个系统消息
        history.append({
            "role": "user",
            "content": content_blocks
        })

    return history


def change_assistant_mode(user_id, user_input, history, client, model, thinking_func, support_image, chat_mode, api,
                          raw_stream):
    print(f"模式：{chat_mode}")
    min_interval = CONFIG.get("min_stream_interval", 0)
    user_message = {"role": "user", "content": user_input}

    try:
        if chat_mode == "情感咨询":
            start_prompt, thinking_content = thinking_func(user_id, user_input, history, client, model, chat_mode)
            thinking_history = {"role": "assistant", "content": thinking_content}
            # history.append(start_prompt)
            history.append(thinking_history)

            # 导入文件
            if support_image:
                full_history = build_full_history_with_references(history, thinking_content, user_input, api,
                                                                  text_dir, image_dir)
            else:
                full_history = build_full_history_with_references_str(history, thinking_content, user_input, api,
                                                                      text_dir, image_dir)
            return full_history

        elif chat_mode == "助手":
            text_file_name = "prompt_template3.txt"
            prompt_path3 = os.path.join(prompt_dir, text_file_name)
            with open(prompt_path3, "r", encoding="utf-8") as f:
                template_str3 = f.read()

            formatted_prompt3 = format_prompt(template_str3, user_input, api, model, support_image)
            start_prompt3 = {"role": "system", "content": formatted_prompt3}

            history.insert(0, start_prompt3)
            history.append(user_message)
            return history

        # ========== 自动对话 ==========
        elif chat_mode == "自动对话":
            rounds = int(user_input)
            # 随机抽两个 client（只在自动对话里）
            if isinstance(client, list):
                if len(client) == 1:
                    client1 = client2 = client[0]
                else:
                    client1, client2 = random.sample(client, 2)
            else:
                client1 = client2 = {"client": client, "model": model}

            #start_prompt, thinking_content = thinking_func(user_id, user_input, history, client1["client"], client1["model"], chat_mode)
            #role1_scene, role2_scene = split_role_scene(thinking_content)

            # 最大重试次数
            max_retry = 3
            retry_count = 0
            while retry_count < max_retry:
                try:
                    start_prompt, thinking_content = thinking_func(
                        user_id, user_input, history, client1["client"], client1["model"], chat_mode
                    )
                    role1_scene, role2_scene = split_role_scene(thinking_content)

                    # 检查是否为空
                    if not role1_scene or not role2_scene:
                        raise ValueError("role1_scene 或 role2_scene 为空，需要重新生成")

                    # 成功生成
                    break

                except Exception as e:
                    retry_count += 1
                    print(f"生成 thinking_content 出错或 role_scene 为空，重试 {retry_count}/{max_retry}，错误：{e}")
                    thinking_content = None
                    role1_scene = role2_scene = None

            else:
                # 超过最大重试次数，从 scenes.json 随机取一个内容
                scenes_path = os.path.join(prompt_dir, "scenes.json")
                with open(scenes_path, "r", encoding="utf-8") as f:
                    scenes_dict = json.load(f)

                if not scenes_dict:
                    raise RuntimeError("scenes.json 文件为空，无法生成默认场景")

                # 随机选一个整数索引的值
                scene_key = random.choice(list(map(int, scenes_dict.keys())))
                fallback_content = scenes_dict[str(scene_key)]

                # 使用 fallback_content 生成 role1_scene, role2_scene
                role1_scene, role2_scene = split_role_scene(fallback_content)
                thinking_content = fallback_content
                print(f"超过重试次数，使用 scenes.json 中的随机内容生成场景，key={scene_key}\n{fallback_content}")

            if (not history) or (len(history) == 1 and history[0].get("role") == "system") or (
                    history and history[-1].get("role") == "assistant"):
                history.append({"role": "user", "content": "你好"})

            def auto_dialogue_generator():
                for _ in range(rounds):
                    # ---------- role1 ----------
                    if history and history[0].get("role") == "system":
                        history.pop(0)

                    text_file_name4 = "prompt_template4.txt"
                    prompt_path4 = os.path.join(prompt_dir, text_file_name4)
                    with open(prompt_path4, "r", encoding="utf-8") as f:
                        template_str4 = f.read()
                    formatted_prompt4 = format_prompt(template_str4, user_input, api, client1["model"], support_image, role1_scene, role2_scene)
                    start_prompt1 = {"role": "system", "content": formatted_prompt4}
                    history.insert(0, start_prompt1)

                    history_copy = copy.deepcopy(history)
                    role1_buffer = ""
                    for chunk in raw_stream(history_copy, role="assistant",
                                            client=client1["client"], model=client1["model"]):
                        yield chunk
                        try:
                            data = json.loads(chunk)
                        except Exception:
                            continue
                        if data.get("type") == "assistant":
                            role1_buffer += data.get("content", "")

                    history.append({"role": "assistant", "content": role1_buffer})

                    # ---------- role2 ----------
                    if history and history[0].get("role") == "system":
                        history.pop(0)

                    text_file_name5 = "prompt_template5.txt"
                    prompt_path5 = os.path.join(prompt_dir, text_file_name5)
                    with open(prompt_path5, "r", encoding="utf-8") as f:
                        template_str5 = f.read()
                    formatted_prompt5 = format_prompt(template_str5, user_input, api, client1["model"], support_image, role1_scene, role2_scene)
                    start_prompt2 = {"role": "system", "content": formatted_prompt5}
                    history.insert(0, start_prompt2)

                    swapped = []
                    for msg in history:
                        if msg.get("role") == "assistant":
                            swapped_role = "user"
                        elif msg.get("role") == "user":
                            swapped_role = "assistant"
                        else:
                            swapped_role = msg.get("role")
                        swapped.append({"role": swapped_role, "content": msg.get("content", "")})

                    role2_buffer = ""
                    for chunk in raw_stream(swapped, role="user",
                                            client=client2["client"], model=client2["model"]):
                        yield chunk
                        try:
                            data = json.loads(chunk)
                        except Exception:
                            continue
                        if data.get("type") == "user":
                            role2_buffer += data.get("content", "")

                    history.append({"role": "user", "content": role2_buffer})

                yield json.dumps({"type": "done"}) + "\n"
                yield json.dumps({"type": "allDone"}) + "\n"
                print(history)

            return stream_with_interval(auto_dialogue_generator(), min_interval)

        else:
            text_file_name = "prompt_template3.txt"
            prompt_path3 = os.path.join(prompt_dir, text_file_name)
            with open(prompt_path3, "r", encoding="utf-8") as f:
                template_str3 = f.read()

            formatted_prompt3 = format_prompt(template_str3, user_input, api, model, support_image)
            start_prompt3 = {"role": "system", "content": formatted_prompt3}

            history.insert(0, start_prompt3)
            history.append(user_message)
            return history

    except Exception as e:
        raise e


def format_prompt(template, user_input, api, model_name, is_support_image, role1_scene="", role2_scene="", kb_dir=text_dir, img_dir=image_dir):
    chat_bot_name = CONFIG.get('chat_bot_name', 'Chat Bot')
    user_name = CONFIG.get('user_name', 'User')
    if api == "baidu":
        api_name = "百度"
    elif api == "KIMI":
        api_name = "KIMI"
    elif api == "ali":
        api_name = "阿里"
    elif api == "tengxun":
        api_name = "腾讯"
    elif api == "deepseek":
        api_name = "Deepseek"
    else:
        api_name = api

    if is_support_image:
        support_image = "已启用"
    else:
        support_image = "已禁用"

    # 获取文件列表并格式化
    def get_file_list(folder_path):
        if not os.path.isdir(folder_path):
            return []

        formatted_files = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                # 匹配 "文件名(关键词...)" + 任意扩展名
                match = re.match(r"^(.*?)\((.*?)\)(\.[^.]+)$", f)
                if match:
                    filename = f"{match.group(1)}{match.group(3)}"  # 保留原扩展名
                    keywords = match.group(2).strip()
                else:
                    filename = f  # 直接保留原文件名和扩展名
                    keywords = "无"
                formatted_files.append(f'文件名：“{filename}”；关键词：“{keywords}”')
        return formatted_files

    # 生成文件列表字符串
    knowledge_base_files = '\n'.join(get_file_list(kb_dir))
    image_base_files = '\n'.join(get_file_list(img_dir))

    # 获取当前日期和时间
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")  # 格式: 2025-08-22
    current_time = now.strftime("%H:%M:%S")  # 格式: 15:13:59

    # 替换固定字段
    formatted = template.replace("{chat_bot_name}", chat_bot_name)
    formatted = formatted.replace("{user_name}", user_name)
    formatted = formatted.replace("{api_name}", api_name)
    formatted = formatted.replace("{model_name}", model_name)
    formatted = formatted.replace("{support_image}", support_image)
    formatted = formatted.replace("{user_input}", user_input)
    formatted = formatted.replace("{knowledge_base}", knowledge_base_files)
    formatted = formatted.replace("{image_base}", image_base_files)
    formatted = formatted.replace("{date}", current_date)
    formatted = formatted.replace("{time}", current_time)
    formatted = formatted.replace("{scene1}", role1_scene)
    formatted = formatted.replace("{scene2}", role2_scene)

    return formatted


def format_chat(messages):
    """
    将对话消息列表格式化为指定文本形式。

    参数:
        messages (list of dict): 每条消息的字典列表，包含 'role' 和 'content'。

    返回:
        str: 格式化后的对话字符串。
    """
    formatted_lines = []
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        # 忽略 system 角色
        if role in ('user', 'assistant'):
            if role == 'user':
                formatted_lines.append(f'user说：“{content}”')
            elif role == 'assistant':
                formatted_lines.append(f'assistant答：“{content}”')
    return '\n'.join(formatted_lines)

def make_random_scene_history():
    """
    构造随机历史记录，user使用prompt_make_scene3.txt，assistant从scenes.json随机抽取，
    最后再添加一条user消息，发送给API，返回模型输出
    """
    # 读取 system prompt
    system_prompt_path = os.path.join(prompt_dir, "prompt_template6.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt_text = f.read()
    system_message = {"role": "system", "content": system_prompt_text}

    # 读取 user prompt
    user_prompt_path = os.path.join(prompt_dir, "prompt_make_scene3.txt")
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_text = f.read()

    # 读取 scenes.json
    scenes_file = os.path.join(prompt_dir, "scenes.json")
    with open(scenes_file, "r", encoding="utf-8") as f:
        scenes_dict = json.load(f)

    # assistant 内容池
    assistant_messages = list(scenes_dict.values())
    random.shuffle(assistant_messages)

    # 随机历史长度（至少1轮 user+assistant）
    max_turns = len(assistant_messages)
    num_turns = random.randint(1, max_turns)

    # 构造历史记录
    history = [ {"role": "system", "content": system_prompt_text} ]
    for i in range(num_turns):
        # user 消息
        history.append({"role": "user", "content": user_prompt_text})
        # assistant 消息
        assistant_content = assistant_messages.pop(0)
        history.append({"role": "assistant", "content": assistant_content})

    # 最后再添加一条 user 消息
    history.append({"role": "user", "content": user_prompt_text})
    return  history

def split_role_scene(scene_text):
    """
    输入完整场景文本，返回两个内容 role_scene1 和 role_scene2，
    每个只包含对应角色的设定
    """
    lines = scene_text.strip().splitlines()
    # 初始化
    chat_scene = ""
    roles = ""
    role1_setting = ""
    role2_setting = ""

    for line in lines:
        line = line.strip()
        if line.startswith("聊天场景") or line.startswith("聊天场景："):
            chat_scene = line.replace("聊天场景：", "").strip()
        elif line.startswith("登场角色") or line.startswith("登场角色："):
            roles = line.replace("登场角色：", "").strip()
        elif line.startswith("角色1设定") or line.startswith("角色1设定："):
            role1_setting = line.replace("角色1设定：", "").strip()
        elif line.startswith("角色2设定") or line.startswith("角色2设定："):
            role2_setting = line.replace("角色2设定：", "").strip()

    # 校验关键信息是否完整
    if not chat_scene or not roles or not role1_setting or not role2_setting:
        return None, None

    # 构造输出
    role_scene1 = f"* **聊天场景**：{chat_scene}\n* **登场角色**：{roles}\n* **你的设定**：{role1_setting}"
    role_scene2 = f"* **聊天场景**：{chat_scene}\n* **登场角色**：{roles}\n* **你的设定**：{role2_setting}"

    return role_scene1, role_scene2

def safe_stream(generator, chat_mode):
    try:
        for chunk in generator:
            yield chunk
    except Exception as e:
        if chat_mode == "自动对话":
            # 自动对话模式：直接把错误发给前端
            print(f"错误：{e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        else:
            # 非自动对话模式：把异常抛给上层，让上层逐个配置重试
            raise

# region APIs
def process_thinking_baidu(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_baidu1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "baidu", model, True)
        start_prompt = {"role": "system", "content": formatted_prompt1}
        history.append(start_prompt)

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_baidu(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream 不动
    def raw_stream(history_input, role="assistant", client=None, model=None, max_retries=5):
        retry_count = 0

        while retry_count < max_retries:
            try:
                #time.sleep(3)  # 每次请求前等待
                response = client.chat.completions.create(
                    model=model,
                    messages=history_input,
                    temperature=0.8,
                    top_p=0.8,
                    stream=True,  # 开启流式
                    extra_body={
                        "penalty_score": 1,
                        "web_search": {"enable": True, "enable_trace": False}
                    }
                )

                got_content = False
                for chunk in response:
                    try:
                        text_part = chunk.choices[0].delta.content
                    except Exception:
                        try:
                            text_part = chunk["choices"][0]["delta"].get("content")
                        except Exception:
                            text_part = None

                    if text_part:
                        got_content = True
                        msg_type = "assistant" if role == "assistant" else "user"
                        yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                        print(text_part, end="")
                print("\n---")

                if got_content:
                    yield json.dumps({"type": "done"}) + "\n"
                    break
                else:
                    retry_count += 1
                    print(f"\n[Warning] No content received. Retrying {retry_count}/{max_retries}...")
                    time.sleep(5)

            except RateLimitError as e:
                retry_count += 1
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                print(f"\n[RateLimitError] Retrying {retry_count}/{max_retries}...")
                time.sleep(5)
            except Exception as e:
                retry_count += 1
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                print(f"\n[Error] Retrying {retry_count}/{max_retries}...")
                time.sleep(5)

        if retry_count >= max_retries:
            yield json.dumps({"type": "error", "message": "Max retries reached, no content received."}) + "\n"

    # ✅ 自动对话模式：一次性建好所有 client
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://qianfan.baidubce.com/v2")
                all_client.append({
                    "client": client,
                    "model": conf["model"]
                })
            except Exception as e:
                print(f"⚠️ 百度配置创建 client 失败: {e}")
                continue

        # 交给 change_assistant_mode，传 all_client
        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_baidu, True, chat_Mode, "baidu", raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个配置尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://qianfan.baidubce.com/v2")
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_baidu, True, chat_Mode, "baidu",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ 百度配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "百度所有配置均失败"}) + "\n"])


def process_thinking_KIMI(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_KIMI1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "KIMI", model, True)
        start_prompt = {"role": "system", "content": formatted_prompt1}
        history.append(start_prompt)

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_KIMI(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream: 保持 KIMI 特性（重试机制 + 延迟）
    def raw_stream(history_input, role="assistant", client=None, model=None, max_retries=5):
        retry_count = 0

        while retry_count < max_retries:
            try:
                time.sleep(3)  # 每次请求前等待
                response = client.chat.completions.create(
                    model=model,
                    messages=history_input,
                    temperature=0.8,
                    top_p=0.8,
                    stream=True,  # 开启流式
                    extra_body={
                        "penalty_score": 1,
                        "web_search": {"enable": True, "enable_trace": False}
                    }
                )

                got_content = False
                for chunk in response:
                    try:
                        text_part = chunk.choices[0].delta.content
                    except Exception:
                        try:
                            text_part = chunk["choices"][0]["delta"].get("content")
                        except Exception:
                            text_part = None

                    if text_part:
                        got_content = True
                        msg_type = "assistant" if role == "assistant" else "user"
                        yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                        print(text_part, end="")
                print("\n---")

                if got_content:
                    yield json.dumps({"type": "done"}) + "\n"
                    break
                else:
                    retry_count += 1
                    print(f"\n[Warning] No content received. Retrying {retry_count}/{max_retries}...")
                    time.sleep(5)

            except RateLimitError as e:
                retry_count += 1
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                print(f"\n[RateLimitError] Retrying {retry_count}/{max_retries}...")
                time.sleep(5)
            except Exception as e:
                retry_count += 1
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                print(f"\n[Error] Retrying {retry_count}/{max_retries}...")
                time.sleep(5)

        if retry_count >= max_retries:
            yield json.dumps({"type": "error", "message": "Max retries reached, no content received."}) + "\n"

    # ✅ 自动对话模式：一次性建好所有 client
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.moonshot.cn/v1")
                all_client.append({"client": client, "model": conf["model"]})
            except Exception as e:
                print(f"⚠️ KIMI配置创建 client 失败: {e}")
                continue

        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_ali, True, chat_Mode, "KIMI", raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个配置尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.moonshot.cn/v1")
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_ali, True, chat_Mode, "KIMI",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ KIMI配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "KIMI所有配置均失败"}) + "\n"])


def process_thinking_ali(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_ali1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        prompt_path2 = os.path.join(prompt_dir, "prompt_template_ali2.txt")
        with open(prompt_path2, "r", encoding="utf-8") as f:
            template_str2 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "ali", model, False)
        formatted_prompt2 = format_prompt(template_str2, user_input, "ali", model, False)

        start_prompt1 = {"role": "system", "content": formatted_prompt1}
        start_prompt2 = {"role": "user", "content": formatted_prompt2}
        history.append(start_prompt1)
        history.append(start_prompt2)

        # 拼接两个格式化后的 prompt
        formatted_prompt = formatted_prompt1 + "\n\n" + formatted_prompt2
        start_prompt = {"role": "system", "content": formatted_prompt}

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_ali(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream 保留原有特性
    def raw_stream(history_input, role="assistant", client=None, model=None):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=history_input,
                temperature=0.8,
                top_p=0.8,
                stream=True,  # 使用流式接口
                extra_body={
                    "penalty_score": 1,
                    "web_search": {
                        "enable": True,
                        "enable_trace": False
                    }
                }
            )

            for chunk in response:
                try:
                    text_part = chunk.choices[0].delta.content
                except Exception:
                    try:
                        text_part = chunk["choices"][0]["delta"].get("content")
                    except Exception:
                        text_part = None

                if text_part:
                    msg_type = "assistant" if role == "assistant" else "user"
                    yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                    print(text_part, end="")
            print("\n---")

            yield json.dumps({"type": "done"}) + "\n"

        except RateLimitError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # ✅ 自动对话模式
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(
                    api_key=conf["api_key"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                all_client.append({
                    "client": client,
                    "model": conf["model"]
                })
            except Exception as e:
                print(f"⚠️ 阿里配置创建 client 失败: {e}")
                continue

        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_ali, False, chat_Mode, "ali", raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(
                    api_key=conf["api_key"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_ali, False, chat_Mode, "ali",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ 阿里配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "阿里所有配置均失败"}) + "\n"])


def process_thinking_tengxun(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_tengxun1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        prompt_path2 = os.path.join(prompt_dir, "prompt_template_tengxun2.txt")
        with open(prompt_path2, "r", encoding="utf-8") as f:
            template_str2 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "tengxun", model, False)
        formatted_prompt2 = format_prompt(template_str2, user_input, "tengxun", model, False)

        start_prompt1 = {"role": "system", "content": formatted_prompt1}
        start_prompt2 = {"role": "user", "content": formatted_prompt2}

        history.insert(0, start_prompt1)
        # history.append(start_prompt1)
        history.append(start_prompt2)

        # 拼接两个格式化后的 prompt
        formatted_prompt = formatted_prompt1 + "\n\n" + formatted_prompt2
        start_prompt = {"role": "user", "content": formatted_prompt}

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_tengxun(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream 不动，只是改成接受 client/model 参数
    def raw_stream(history_input, role="assistant", client=None, model=None):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=history_input,
                temperature=0.8,
                top_p=0.8,
                stream=True,  # 开启流式
                extra_body={
                    "penalty_score": 1,
                    "web_search": {
                        "enable": True,
                        "enable_trace": False
                    }
                }
            )

            for chunk in response:
                try:
                    text_part = chunk.choices[0].delta.content
                except Exception:
                    try:
                        text_part = chunk["choices"][0]["delta"].get("content")
                    except Exception:
                        text_part = None

                if text_part:
                    msg_type = "assistant" if role == "assistant" else "user"
                    yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                    print(text_part, end="")
            print("\n---")
            yield json.dumps({"type": "done"}) + "\n"

        except RateLimitError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # ✅ 自动对话模式：一次性建好所有 client
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.hunyuan.cloud.tencent.com/v1")
                all_client.append({
                    "client": client,
                    "model": conf["model"]
                })
            except Exception as e:
                print(f"⚠️ 腾讯配置创建 client 失败: {e}")
                continue

        # 注意 process_thinking_tengxun，False 保留
        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_tengxun, False, chat_Mode, "tengxun",
            raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个配置尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.hunyuan.cloud.tencent.com/v1")
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_tengxun, False, chat_Mode, "tengxun",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ 腾讯配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "腾讯所有配置均失败"}) + "\n"])


def process_thinking_deepseek(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_deepseek1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "deepseek", model, True)
        start_prompt = {"role": "system", "content": formatted_prompt1}
        history.append(start_prompt)

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_deepseek(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream: role 决定 type 字段（assistant -> "text", user -> "user"）
    def raw_stream(history_input, role="assistant", client=None, model=None):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=history_input,
                temperature=0.8,
                top_p=0.8,
                stream=True,  # 开启流式
                extra_body={
                    "penalty_score": 1,
                    "web_search": {"enable": True, "enable_trace": False}
                }
            )

            for chunk in response:
                try:
                    text_part = chunk.choices[0].delta.content
                except Exception:
                    try:
                        text_part = chunk["choices"][0]["delta"].get("content")
                    except Exception:
                        text_part = None

                if text_part:
                    msg_type = "assistant" if role == "assistant" else "user"
                    yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                    print(text_part, end="")
            print("\n---")

            # 流结束
            yield json.dumps({"type": "done"}) + "\n"

        except RateLimitError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # ✅ 自动对话模式：一次性建好所有 client
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.deepseek.com")
                all_client.append({
                    "client": client,
                    "model": conf["model"]
                })
            except Exception as e:
                print(f"⚠️ DeepSeek 配置创建 client 失败: {e}")
                continue

        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_deepseek, True, chat_Mode, "deepseek",
            raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个配置尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://api.deepseek.com")
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_deepseek, True, chat_Mode, "deepseek",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ DeepSeek 配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "DeepSeek 所有配置均失败"}) + "\n"])


def process_thinking_lizhi(user_id, user_input, history, client, model, chat_mode):
    start_prompt = None
    if chat_mode == "情感咨询":
        prompt_path1 = os.path.join(prompt_dir, "prompt_template_lizhi1.txt")
        with open(prompt_path1, "r", encoding="utf-8") as f:
            template_str1 = f.read()

        formatted_prompt1 = format_prompt(template_str1, user_input, "lizhi", model, True)
        start_prompt = {"role": "system", "content": formatted_prompt1}
        history.append(start_prompt)

    elif chat_mode == "自动对话":
        history = make_random_scene_history()

    #print(client)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body={
                "penalty_score": 1,
                "web_search": {
                    "enable": True,
                    "enable_trace": False
                }
            }
        )
        assistant_text = response.choices[0].message.content
        return start_prompt, assistant_text

    except RateLimitError as e:
        raise e


def process_user_input_lizhi(user_id, user_input, history, conf_list, chat_Mode):
    min_interval = CONFIG.get("min_stream_interval", 0)

    # raw_stream 不动
    def raw_stream(history_input, role="assistant", client=None, model=None):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=history_input,
                temperature=0.8,
                top_p=0.8,
                stream=True,
                extra_body={
                    "penalty_score": 1,
                    "web_search": {"enable": True, "enable_trace": False}
                }
            )

            for chunk in response:
                try:
                    text_part = chunk.choices[0].delta.content
                except Exception:
                    try:
                        text_part = chunk["choices"][0]["delta"].get("content")
                    except Exception:
                        text_part = None

                if text_part:
                    msg_type = "assistant" if role == "assistant" else "user"
                    yield json.dumps({"type": msg_type, "content": text_part}) + "\n"
                    print(text_part, end="")
            print("\n---")
            yield json.dumps({"type": "done"}) + "\n"

        except RateLimitError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # ✅ 自动对话模式：一次性建好所有 client
    if chat_Mode == "自动对话":
        all_client = []
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://ark.cn-beijing.volces.com/api/v3")
                all_client.append({
                    "client": client,
                    "model": conf["model"]
                })
            except Exception as e:
                print(f"⚠️ 荔枝配置创建 client 失败: {e}")
                continue

        full_history = change_assistant_mode(
            user_id, user_input, history, all_client, None, process_thinking_lizhi, True, chat_Mode, "lizhi", raw_stream
        )
        return safe_stream(full_history, chat_Mode)

    # ✅ 非自动对话模式：逐个配置尝试
    else:
        for conf in conf_list:
            try:
                client = OpenAI(api_key=conf["api_key"], base_url="https://ark.cn-beijing.volces.com/api/v3")
                model = conf["model"]

                full_history = change_assistant_mode(
                    user_id, user_input, history, client, model, process_thinking_lizhi, True, chat_Mode, "lizhi",
                    raw_stream
                )
                return safe_stream(stream_with_interval(raw_stream(full_history, client=client, model=model), min_interval), chat_Mode)

            except Exception as e:
                print(f"⚠️ 荔枝配置 {conf.get('api_key', '')[:6]}... 失败，切换下一个: {e}")
                continue

        # 如果所有配置都失败
        return iter([json.dumps({"type": "error", "message": "荔枝所有配置均失败"}) + "\n"])


# endregion

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)