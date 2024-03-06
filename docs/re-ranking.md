

Models are better at using relevant information that occurs at the very begging(primacy bias) or end of it's input context (recency bias), and performance degrades significantly when models have to use information located in the middle of the input context. 

## Why Rerank?
Reranking is pivotal in RAG systems, addressing critical limitations observed in initial retrieval phases, particularly within the context of semantic search. The empirical findings from the paper "Lost in the Middle: How Language Models Use Long Contexts"[1] underscore the performance degradation of language models when relevant information is positioned in the middle of long input contexts, signifying a need for effective reranking mechanisms. By employing reranking, systems can counteract the inherent weaknesses of language models in handling extensive and complex contexts, ensuring that the most contextually pertinent information, irrespective of its original position in the retrieved set, is elevated. This strategy not only enhances the precision of search outcomes by leveraging semantic embeddings but also substantially improves the model's ability to synthesize and utilize relevant information from a vast pool of candidates, thereby optimizing the performance of RAG tasks in processing intricate queries and extensive documents.

## Document Results/Experiments
(todo) document flash reranking experiment
(todo) document mxbai experiment

Todos:

[ ] [LangChain Long-Context Reorder](https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder)

[ ] 

## Reference
1) [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172.pdf). Liu et all. 
