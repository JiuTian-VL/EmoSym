import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

while True:
    try:
        snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", local_dir='/data1/Models/Qwen2.5-0.5B-Instruct',
                     local_dir_use_symlinks=False, etag_timeout=500)
    except Exception as e :
        print(e)
        # time.sleep(5)
    else:
        print('下载完成')
        break
