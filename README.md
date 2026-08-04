## Chunk

Um **chunk** é um pequeno segmento de um documento. Em vez de armazenar um ficheiro inteiro como uma única unidade, o texto é dividido em partes menores, facilitando a pesquisa e reduzindo a quantidade de informação enviada ao modelo de linguagem.

Por exemplo, um documento com várias páginas pode ser dividido em dezenas ou centenas de chunks. Quando o utilizador faz uma pergunta, apenas os chunks mais relevantes são recuperados.

---

## Embedding

Um **embedding** é uma representação numérica do significado de um texto. Um modelo de embeddings converte cada chunk (e cada pergunta do utilizador) num vetor de números que captura o seu conteúdo semântico.

Graças a estes vetores, é possível comparar textos pelo seu significado e não apenas pelas palavras que contêm. Isto permite encontrar informação relevante mesmo quando a pergunta utiliza palavras diferentes das presentes no documento.

---

## ChromaDB

O **ChromaDB** é uma base de dados vetorial utilizada para armazenar os embeddings dos documentos. Além dos vetores, também pode guardar o texto original e metadados, como o nome do ficheiro ou a página de origem.

Quando uma pergunta é feita, o ChromaDB compara o embedding da pergunta com os embeddings armazenados e devolve os chunks semanticamente mais semelhantes.

---

## Modelo de Embeddings

O **modelo de embeddings** é responsável por transformar texto em vetores numéricos. Este modelo é utilizado tanto durante a indexação dos documentos como durante a pesquisa.

* **Indexação:** cada chunk é convertido num embedding e armazenado no ChromaDB.
* **Pesquisa:** a pergunta do utilizador é convertida num embedding para que possa ser comparada com os embeddings armazenados.

Este modelo não gera respostas; apenas cria representações vetoriais dos textos.

---

## LLM (Large Language Model)

O **LLM** é o modelo responsável por gerar a resposta final ao utilizador. Depois de o ChromaDB devolver os chunks mais relevantes, estes são fornecidos ao LLM como contexto.

Com base na pergunta e na informação recuperada, o LLM produz uma resposta natural e fundamentada nos documentos fornecidos.

---

## Fluxo de funcionamento do RAG

O funcionamento de um sistema RAG pode ser resumido em duas fases:

### 1. Indexação

1. Os documentos são divididos em **chunks**.
2. Cada chunk é convertido num **embedding** através de um modelo de embeddings.
3. Os embeddings, juntamente com os respetivos textos e metadados, são armazenados no **ChromaDB**.

### 2. Pesquisa e geração de resposta

1. O utilizador faz uma pergunta.
2. A pergunta é convertida num **embedding** pelo mesmo modelo de embeddings.
3. O **ChromaDB** procura os chunks cujos embeddings são mais semelhantes ao da pergunta.
4. Os chunks encontrados são enviados para o **LLM**.
5. O **LLM** utiliza esse contexto para gerar a resposta final.
