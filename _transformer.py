from transformers import AutoModel

model = AutoModel.from_pretrained('facebook/opt-1.3b', trust_remote_code=True)
print(model)
print(model.decoder.layers)

