


def chunk_text(text, chunk_size=300, overlap=50): ## splitting text into chunks 
    words = text.split()
    chunks = []                 ## collect finished chunks (there's a 50 word overlap between neighboring chunks)
    step = chunk_size - overlap 
    
    for i in range(0, len(words), step): ## start positioning (0,250, 500)
        chunk = words[i:i + chunk_size] 
        chunks.append(" ".join(chunk))


    return chunks

if __name__=="__main__":
    sample = "word " * 700
    chunks = chunk_text(sample)
    print(f"number of chunks: {len(chunks)}")

    for i, c in enumerate(chunks):
        print(f"chunks {i}:{len(c.split())} words")
