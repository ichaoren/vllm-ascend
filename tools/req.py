import sys
import requests
import json
import time

def main():
    # 接口地址
    url = "http://141.61.33.12:8888/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # 解析入参：python test_api.py prompt.txt [repeat_times]
    repeat_times = 1
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    elif len(sys.argv) == 3:
        file_path = sys.argv[1]
        try:
            repeat_times = int(sys.argv[2])
            if repeat_times < 1:
                print("错误：重复次数必须为大于等于1的正整数")
                return
        except ValueError:
            print("错误：第二个参数重复次数必须是整数")
            return
    else:
        print("使用方式：python test_api.py 你的prompt文件.txt [重复次数]")
        print("示例：")
        print("  python test_api.py input.txt")
        print("  python test_api.py input.txt 3")
        return

    # 读取文件全部内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return
    except Exception as e:
        print(f"读取文件失败：{e}")
        return

    # 将文本重复N遍
    prompt_content = prompt_content * repeat_times
    print(f"已将输入文本重复 {repeat_times} 次，总长度：{len(prompt_content)}")

    # 构造请求体
    payload = {
        "model": "ds",
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "stream": False,
        "ignore_eos": False,
        "max_tokens": 1,
        "temperature": 0
    }

    # 发送POST请求，统计耗时
    try:
        start = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload)
        end = time.perf_counter()
        cost_ms = (end - start) * 1000

        print("=" * 50)
        print(f"状态码：{resp.status_code}")
        print(f"请求耗时：{cost_ms:.2f} ms")
        print("=" * 50)
        # 格式化输出返回JSON
        #print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"接口请求异常：{e}")

if __name__ == "__main__":
    main()
