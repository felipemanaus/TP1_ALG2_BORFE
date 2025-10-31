import os
import re
import json
from collections import defaultdict
import math
from compact_trie import CompactTrie, TrieNode 

class Indexer:
    """
    Orquestra a indexação:
    Lê o corpus, cria a Trie (índice), o mapa de docs e as estatísticas (Z-score).
    """
    
    def __init__(self, corpus_path: str, trie_file="inverted_index.txt", map_file="doc_id_map.json", stats_file="global_stats.json"):
        self.corpus_path = corpus_path
        self.trie_file = trie_file
        self.map_file = map_file
        self.stats_file = stats_file
        
        self.trie = CompactTrie()
        self.doc_map = {}
        # Guarda dados para z-score: {termo: {'mu': X, 'sigma': Y, 'df': Z}}
        self.global_stats = {} 
        self.total_docs = 0 

    def _load_or_create_index_data(self):
        """Tenta carregar os dados (Trie, mapa, stats) do disco."""
        
        if self.trie.load_from_file(self.trie_file):
            print(f"Índice carregado de {self.trie_file}.")
            
            try:
                with open(self.map_file, 'r', encoding='utf-8') as f:
                    doc_map_str = json.load(f)
                    self.doc_map = {int(k): v for k, v in doc_map_str.items()}
                    self.total_docs = len(self.doc_map)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                print("Mapeamento não encontrado ou corrompido.")
            
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.global_stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                print("Estatísticas não encontradas ou corrompidas.")

            # Retorna True se tudo foi carregado
            if self.doc_map and self.global_stats:
                return True
        
        print("Iniciando indexação a partir do zero.")
        return False

    def _tokenize_and_calculate_tf(self, text):
        """Limpa o texto, quebra em tokens e conta a Frequência (TF)."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9&-]+', text)
        
        term_frequency = defaultdict(int)
        for token in tokens:
            term_frequency[token] += 1
            
        return term_frequency

    def index_corpus(self):
        """Orquestra a indexação (ou carrega se já existir)."""
        
        if self._load_or_create_index_data():
            return # Já estava pronto
        
        doc_id_counter = 1
        
        # Passagem 1: Ler arquivos, construir Trie e coletar stats brutos.
        raw_stats = {} 

        print("Passagem 1: Lendo documentos e construindo a Trie...")
        
        for root, _, files in os.walk(self.corpus_path):
            for file_name in files:
                if file_name.endswith('.txt'):
                    
                    file_path_full = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path_full, self.corpus_path)
                    
                    doc_id = doc_id_counter
                    self.doc_map[doc_id] = relative_path
                    doc_id_counter += 1
                    
                    try:
                        with open(file_path_full, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        term_frequencies = self._tokenize_and_calculate_tf(content)
                        
                        for term, tf in term_frequencies.items():
                            
                            # 1. Insere na Trie
                            self.trie.insert(term, doc_id, tf)
                            
                            # 2. Coleta dados para Z-score
                            if term not in raw_stats:
                                raw_stats[term] = {'sum_tf': 0, 'sum_tf2': 0, 'df': 0}

                            raw_stats[term]['df'] += 1 
                            raw_stats[term]['sum_tf'] += tf
                            raw_stats[term]['sum_tf2'] += (tf ** 2)
                            
                        self.total_docs = doc_id
                        if doc_id % 200 == 0:
                            print(f"Indexados {doc_id} documentos...")
                            
                    except Exception as e:
                        print(f"Erro ao processar o arquivo {file_path_full}: {e}")

        # Passagem 2: Calcular stats finais (Z-score) e salvar tudo.
        self._calculate_and_save_stats(raw_stats)
        print("Módulo de Indexação encerrado.")

    def _calculate_and_save_stats(self, raw_stats):
        """Calcula Média (mu) e Desvio-Padrão (sigma) e salva tudo em disco."""
        
        num_docs = self.total_docs
        print(f"\nCalculando estatísticas finais para {num_docs} documentos...")
        
        final_stats = {}
        for term, data in raw_stats.items():
            
            sum_tf = data['sum_tf']
            sum_tf2 = data['sum_tf2']
            df = data['df'] # Document Frequency
            
            # Média (Mu)
            mu = sum_tf / df if df > 0 else 0 
            
            # Desvio-Padrão (Sigma)
            variance = (sum_tf2 / df) - (mu ** 2) if df > 0 else 0
            sigma = math.sqrt(variance) if variance >= 0 else 0
            
            final_stats[term] = {
                'mu': mu,
                'sigma': sigma,
                'df': df
            }
        
        self.global_stats = final_stats

        # --- Salvamento ---
        
        # 1. Salva a Trie
        self.trie.save_to_file(self.trie_file)
        print(f"Índice invertido salvo em: {self.trie_file}")
        
        # 2. Salva o Mapeamento
        with open(self.map_file, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.doc_map.items()}, f, indent=4)
        print(f"Mapeamento salvo em: {self.map_file}")

        # 3. Salva as Estatísticas
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.global_stats, f, indent=4)
        print(f"Estatísticas Z-score salvas em: {self.stats_file}")

# --- Execução direta ---
if __name__ == '__main__':
    CORPUS_FOLDER = "bbc" 
    
    if not os.path.exists(CORPUS_FOLDER):
        print(f"ERRO: A pasta do corpus '{CORPUS_FOLDER}' não foi encontrada.")
    else:
        print("--- INICIANDO PROCESSO DE INDEXAÇÃO MANUAL ---")
        indexer = Indexer(corpus_path=CORPUS_FOLDER)
        indexer.index_corpus()
        print("\n--- PROCESSO DE INDEXAÇÃO MANUAL CONCLUÍDO ---")