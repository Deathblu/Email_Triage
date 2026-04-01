from huggingface_hub import HfApi,login 

login(token="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")
api = HfApi()
api.upload_file(
    path_or_fileobj='inference.py',
    path_in_repo='inference.py',
    repo_id='Deathblue1306/email-triage-env',
    repo_type='space'
)
print('uploaded')   