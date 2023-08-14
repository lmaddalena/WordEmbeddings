from gensim.models import KeyedVectors

model = KeyedVectors.load("data/text8-word2vec.bin")
word_vectors = model.wv

words_to_index = word_vectors.key_to_index
words = list(words_to_index.keys())

print([words[i] for i in range(20)])
print("\n")

assert("king" in words)

# Check the "most similar words", using the default "cosine similarity" measure.
print('Check the "most similar words", using the default "cosine similarity" measure.')

result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")

print("\n")


print('Check the "most similar words", using the default "cosine similarity" measure.')
result = word_vectors.most_similar(positive=['king'])

for i in range(4):
	most_similar_key, similarity = result[i]
	print(f"{most_similar_key}: {similarity:.4f}")

print("\n")

# similar by word "cat"
print('similar by word "cat"')

result = word_vectors.similar_by_word("cat")
for i in range(4):
	most_similar_key, similarity = result[i]
	print(f"{most_similar_key}: {similarity:.4f}")

print("\n")
