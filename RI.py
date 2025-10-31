import os
import math
import json
from collections import deque
from compact_trie import CompactTrie 

# Define a precedência dos operadores para o Shunting-Yard
OP_PRECEDENCE = {
    'OR': 1,
    'AND': 2,
    '(': 0,
    ')': 0
}

class InformationRetriever:
    """
    Processa consultas booleanas e ranqueia os resultados por Z-score.
    """
    def __init__(self, trie_file="inverted_index.txt", stats_file="global_stats.json"):
        self.trie = CompactTrie()
        self.global_stats = {}
        self.is_ready = False # Flag que indica se os dados carregaram

        if self._load_data(trie_file, stats_file):
            self.is_ready = True
        
    def _load_data(self, trie_file, stats_file):
        """Carrega a Trie (índice) e as estatísticas (Z-score) do disco."""
        
        if not self.trie.load_from_file(trie_file):
            print(f"ERRO: Falha ao carregar a Trie do arquivo {trie_file}.")
            return False
            
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.global_stats = json.load(f)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"ERRO: Falha ao carregar as estatísticas de Z-score do arquivo {stats_file}.")
            return False

    # ====================================================================
    # LÓGICA BOOLEANA (SHUNTING-YARD E AVALIAÇÃO RPN)
    # ====================================================================

    def _tokenize_query(self, query: str) -> list:
        """Quebra a string da consulta em tokens (termos e operadores)."""
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        tokens = query.split()
        
        processed_tokens = []
        for token in tokens:
            if token.strip():
                if token in OP_PRECEDENCE:
                    processed_tokens.append(token) # Operadores (AND, OR, () )
                else:
                    processed_tokens.append(token.lower()) # Termos
        return processed_tokens

    def _to_rpn(self, tokens: list) -> list:
        """Converte tokens para Notação Polonesa Reversa (RPN) via Shunting-Yard."""
        output = []
        operator_stack = deque()

        for token in tokens:
            if token not in OP_PRECEDENCE:
                output.append(token) # Termo
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop() # Descarta o '('
            else:
                # Operador (AND ou OR)
                while (operator_stack and operator_stack[-1] != '(' and 
                       OP_PRECEDENCE.get(operator_stack[-1], 0) >= OP_PRECEDENCE[token]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)

        while operator_stack:
            output.append(operator_stack.pop())
            
        return output

    def _evaluate_rpn(self, rpn_tokens: list) -> set:
        """Avalia a consulta RPN e retorna o conjunto de DocIDs."""
        operand_stack = deque()

        for token in rpn_tokens:
            if token not in OP_PRECEDENCE:
                # É um termo: busca os DocIDs na Trie
                index_list = self.trie.find(token)
                doc_ids = {doc_id for doc_id, tf in index_list}
                operand_stack.append(doc_ids)
            
            elif token == 'AND':
                # Interseção
                if len(operand_stack) < 2: raise ValueError("Consulta AND mal formada.")
                set2 = operand_stack.pop()
                set1 = operand_stack.pop()
                operand_stack.append(set1.intersection(set2))
                
            elif token == 'OR':
                # União
                if len(operand_stack) < 2: raise ValueError("Consulta OR mal formada.")
                set2 = operand_stack.pop()
                set1 = operand_stack.pop()
                operand_stack.append(set1.union(set2))

        if len(operand_stack) != 1:
            raise ValueError("Consulta Booleana inválida.")
            
        return operand_stack.pop()

    # ====================================================================
    # RANQUEAMENTO POR Z-SCORE
    # ====================================================================
    
    def _calculate_z_score(self, tf: int, term: str) -> float:
        """Calcula o Z-score de um termo (TF) usando as estatísticas globais."""
        stats = self.global_stats.get(term)
        if not stats:
            return 0.0
        
        mu = stats['mu']
        sigma = stats['sigma']
        
        if sigma <= 0:
            return 1.0 if tf > mu else 0.0
        
        return (tf - mu) / sigma

    def _rank_results(self, doc_ids: set, query_terms: set) -> list:
        """Calcula a relevância (média dos Z-scores) e ordena os DocIDs."""
        ranked_docs = [] # Lista de (relevância, doc_id)

        for doc_id in doc_ids:
            total_z_score = 0.0
            term_count = 0
            
            for term in query_terms:
                # 1. Acha o TF do termo neste doc
                index_list = self.trie.find(term)
                tf = next((t for d, t in index_list if d == doc_id), 0)
                
                if tf > 0:
                    # 2. Calcula o Z-score
                    z_score = self._calculate_z_score(tf, term)
                    total_z_score += z_score
                    term_count += 1
            
            if term_count > 0:
                # 3. Relevância = Média dos Z-scores
                relevance = total_z_score / term_count
                ranked_docs.append((relevance, doc_id))

        # Ordena pela relevância (maior primeiro)
        ranked_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc_id for relevance, doc_id in ranked_docs]

    # ====================================================================
    # FUNÇÃO PRINCIPAL
    # ====================================================================

    def search(self, query: str) -> list:
        """Executa a busca booleana e ranqueada."""
        if not self.is_ready:
            return []

        try:
            # 1. Processa a query
            tokens = self._tokenize_query(query)
            query_terms = {t for t in tokens if t not in OP_PRECEDENCE}
            
            # 2. Filtro Booleano
            rpn_tokens = self._to_rpn(tokens)
            matching_doc_ids = self._evaluate_rpn(rpn_tokens)
            
            if not matching_doc_ids:
                return []
                
            # 3. Ranqueamento
            return self._rank_results(matching_doc_ids, query_terms)
        
        except ValueError as e:
            print(f"ERRO DE CONSULTA: {e}")
            return []

