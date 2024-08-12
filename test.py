import os

from Bpe import RegexTokenizer


# 1、实例化tokenizer
tokenizer = RegexTokenizer(dropout=None)

# 2、导入文本数据
text = ""
name_list = os.listdir("D:\code\\textcrawl\datasets\h_hub\gzdqy")
for name in name_list:
    with open(f"D:\code\\textcrawl\datasets\h_hub\gzdqy\\{name}", "r", encoding="utf-8") as f:
        text += f.read().strip() + "\n"

# 3、训练tokenizer
tokenizer.train(text, vocab_size=1000, verbose=True)

# 4、注册special_tokens
tokenizer.register_special_tokens({
    "<|startoftext|>": 1000,
    "<|endoftext|>": 1001,
    "<|padding|>": 1002
})
print(tokenizer.merges)
print(tokenizer.vocab)
print(tokenizer.special_tokens)
print(tokenizer.pattern)
print("===================================\n")

# 5、保存训练结果
tokenizer.save("LLMBpe")

# 6、加载训练结果
tokenizer = RegexTokenizer()
tokenizer.load("LLMBpe.model")
print(tokenizer.merges)
print(tokenizer.vocab)
print(tokenizer.special_tokens)
print(tokenizer.pattern)
print("===================================\n")

# 7、编码
with open("D:\code\\textcrawl\datasets\h_hub\gzdqy\\3.txt", "r", encoding="utf-8") as f:
    en_text = "<|startoftext|>" + f.read().strip() + "<|endoftext|>" + "<|padding|>" + "<|padding|>"
ids = tokenizer.encode(en_text)
print(len(en_text))
print(len(ids))

# 8、解码
de_text = tokenizer.decode(ids)
print(en_text == de_text)



