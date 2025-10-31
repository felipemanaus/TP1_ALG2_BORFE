class TrieNode:
    """ Representa um nó na Árvore Trie Compacta. """
    def __init__(self):
        self.label = ""
        # Dicionário onde a chave é o primeiro caractere do rótulo do filho
        self.children = {}
        self.is_terminal = False
        # Lista de (DocID, Frequência)
        self.inverted_index = []

class CompactTrie:
    """ Implementação da Árvore Trie Compacta (Radix Tree). """
    def __init__(self):
        self.root = TrieNode()
        
    def _find_mismatch_point(self, word, label):
        """ Retorna o comprimento do prefixo comum entre a palavra e o rótulo. """
        i = 0
        min_len = min(len(word), len(label))
        while i < min_len and word[i] == label[i]:
            i += 1
        return i

    def insert(self, word: str, doc_id: int, frequency: int):
        """ Insere uma palavra na Trie e atualiza seu índice invertido. """
        current_node = self.root
        remaining_word = word
        
        while remaining_word:
            char = remaining_word[0]
            
            if char not in current_node.children:
                # Caso 1: Não há caminho. (Inserção simples)
                new_node = TrieNode()
                new_node.label = remaining_word
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                current_node.children[char] = new_node
                return
            
            child_node = current_node.children[char]
            child_label = child_node.label
            
            mismatch_idx = self._find_mismatch_point(remaining_word, child_label)
            
            # ----------------------------------------------------
            # A. Match Exato: (Palavra == Rótulo do Filho)
            # ----------------------------------------------------
            if mismatch_idx == len(remaining_word) and mismatch_idx == len(child_label):
                child_node.is_terminal = True 
                child_node.inverted_index.append((doc_id, frequency))
                return
            
            # ----------------------------------------------------
            # B. Palavra é Prefixo de Rótulo: (Palavra é mais curta)
            # ----------------------------------------------------
            elif mismatch_idx == len(remaining_word):
                # Quebra o nó filho em dois: (ex: "carro" vira "car" -> "ro")
                
                # 1. Nó para a palavra (prefixo)
                new_node = TrieNode()
                new_node.label = remaining_word
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                # 2. Atualiza o nó antigo (resto)
                remaining_label = child_label[mismatch_idx:]
                child_node.label = remaining_label
                
                # 3. Liga o nó antigo ao novo
                new_node.children[remaining_label[0]] = child_node
                
                # 4. Liga o novo nó ao pai
                current_node.children[char] = new_node
                return
            
            # ----------------------------------------------------
            # C. Rótulo é Prefixo de Palavra: (Palavra é mais longa)
            # ----------------------------------------------------
            elif mismatch_idx == len(child_label):
                # Continua a busca a partir do nó filho
                current_node = child_node
                remaining_word = remaining_word[mismatch_idx:]
                
            # ----------------------------------------------------
            # D. Divergência e Split (Prefixos comuns e restos)
            # ----------------------------------------------------
            elif 0 < mismatch_idx < len(child_label):
                # (Ex: Inserir "carga" | Nó é "carro" -> "car" vira pai de "ga" e "ro")
                
                # 1. Nó de divisão (prefixo comum)
                split_node = TrieNode()
                split_node.label = child_label[:mismatch_idx]
                
                # 2. Atualiza o rótulo do nó filho antigo
                child_node.label = child_label[mismatch_idx:]
                
                # 3. Liga o nó filho ao nó de divisão
                split_node.children[child_node.label[0]] = child_node
                
                # 4. O restante da nova palavra vira um novo nó filho
                new_word_part = remaining_word[mismatch_idx:]
                new_node = TrieNode()
                new_node.label = new_word_part
                new_node.is_terminal = True
                new_node.inverted_index.append((doc_id, frequency))
                
                split_node.children[new_word_part[0]] = new_node
                
                # 5. Liga o nó de divisão ao nó atual
                current_node.children[char] = split_node
                return
    
    def find(self, word: str) -> list:
        """
        Busca uma palavra na Trie Compacta.
        Retorna o índice invertido (lista) ou [] se não encontrar.
        """
        current_node = self.root
        remaining_word = word
        
        while remaining_word:
            char = remaining_word[0]
            
            if char not in current_node.children:
                return []
            
            child_node = current_node.children[char]
            child_label = child_node.label
            
            mismatch_idx = self._find_mismatch_point(remaining_word, child_label)
            
            if mismatch_idx == len(remaining_word):
                if mismatch_idx == len(child_label):
                    # Match exato
                    current_node = child_node
                    remaining_word = "" 
                    break
                
                return [] # Palavra é prefixo de um rótulo, mas não é nó

            elif mismatch_idx == len(child_label):
                # Rótulo é prefixo da palavra, continua descendo
                current_node = child_node
                remaining_word = remaining_word[mismatch_idx:]
                
            else: 
                # Divergência
                return []
        
        if current_node.is_terminal:
            return current_node.inverted_index
        else:
            return []
        
    def pre_order_serialize(self, node: TrieNode, file_handler):
        """ Função auxiliar recursiva para serializar em Pré-Ordem. """
        index_str = ";".join([f"{doc},{freq}" for doc, freq in node.inverted_index])
        
        # Formato: "label|is_terminal|num_children|inverted_index_string"
        line = f"{node.label}|{1 if node.is_terminal else 0}|{len(node.children)}|{index_str}\n"
        
        file_handler.write(line)
        
        for char in sorted(node.children.keys()):
            child_node = node.children[char]
            self.pre_order_serialize(child_node, file_handler)
            
    def save_to_file(self, filename: str):
        """ Persiste a CompactTrie em disco. """
        print(f"Salvando índice para {filename}...")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                self.pre_order_serialize(self.root, f)
            print("Salvamento concluído.")
        except Exception as e:
            print(f"Erro ao salvar o índice: {e}")
            
    def load_from_file(self, filename: str):
        """ Carrega a CompactTrie do disco, reconstruindo a estrutura. """
        print(f"Carregando índice de {filename}...")
        
        # Pilha para reconstrução (nó_pai, num_filhos_restantes)
        stack = [] 
        
        self.root = TrieNode()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                
                # 1. Processa a raiz (primeira linha)
                first_line = f.readline()
                if not first_line:
                    return False # Arquivo vazio
                
                label, is_terminal_str, num_children_str, index_str = first_line.strip().split('|')
                
                self.root.label = label 
                self.root.is_terminal = is_terminal_str == '1'
                num_children = int(num_children_str)
                
                if index_str:
                    for item in index_str.split(';'):
                        doc_id, freq = map(int, item.split(','))
                        self.root.inverted_index.append((doc_id, freq))
                
                if num_children > 0:
                    stack.append((self.root, num_children))

                # 2. Processa os descendentes
                for line in f:
                    if not stack:
                        break 
                        
                    parent_node, remaining_children = stack[-1] 
                    
                    label, is_terminal_str, num_children_str, index_str = line.strip().split('|')
                    
                    new_node = TrieNode()
                    new_node.label = label
                    new_node.is_terminal = is_terminal_str == '1'
                    new_node.children = {}
                    new_node.inverted_index = []
                    num_children = int(num_children_str)
                    
                    if index_str:
                        for item in index_str.split(';'):
                            doc_id, freq = map(int, item.split(','))
                            new_node.inverted_index.append((doc_id, freq))

                    parent_node.children[new_node.label[0]] = new_node
                    
                    remaining_children -= 1
                    if remaining_children == 0:
                        stack.pop()
                    else:
                        stack[-1] = (parent_node, remaining_children)
                        
                    if num_children > 0:
                        stack.append((new_node, num_children))
                        
            print("Carregamento concluído.")
            return True
            
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Nenhuma Trie carregada.")
            return False
        except Exception as e:
            print(f"Erro ao carregar ou desserializar o índice: {e}")
            return False