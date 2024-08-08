from Bpe import RegexTokenizer


# 1、实例化tokenizer
tokenizer = RegexTokenizer()

# 2、导入文本数据
with open("D:\code\\textcrawl\datasets\gzdqy\\1.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# 3、训练tokenizer
tokenizer.train(text, vocab_size=300, verbose=True)

# 4、注册special_tokens
tokenizer.register_special_tokens({
    "<|startoftext|>": 300,
    "<|endoftext|>": 301,
    "<|padding|>": 302
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
with open("D:\code\\textcrawl\datasets\gnszjlb\\3.txt", "r", encoding="utf-8") as f:
    en_text = "<|startoftext|>" + f.read().strip() + "<|endoftext|>" + "<|padding|>" + "<|padding|>"
ids = tokenizer.encode(en_text)
print(len(ids))

# 8、解码
de_text = tokenizer.decode(ids)
print(en_text == de_text)



