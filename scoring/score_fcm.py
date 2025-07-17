import pandas as pd
import json
import numpy as np
import ast
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel



class ScoreCalculator:
    cache = {}
    def __init__(self, threshold,model_name,data,tp_scale=1,pp_scale=1):#,fp_scale=1,fn_scale=1):
        self.model_name=model_name
        self.data=data
        self.threshold = threshold
        self.tp_scale = tp_scale
        self.pp_scale=pp_scale
        # self.fp_scale=fp_scale
        # self.fn_scale=fn_scale
        
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
        '''
        best params:
        QWEN 0.6B
            threshold: .6 
            tp_scale: 1.0
            pp_scale: 1.1
        QWEN 8B
            threshold: .5
            tp_scale: .4
            pp_scale: .1
        '''


        self.task_instruction = "Find concepts that represent variables capable of the same or similar quantitative or qualitative changes."
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = self.model.to(self.device)

    def last_token_pool(self,last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def embed_and_score(self,queries, documents):

        clean_queries = [str(q).strip() for q in queries if q is not None and str(q).strip()]
        clean_documents = [str(d).strip() for d in documents if d is not None and str(d).strip()]

        if not clean_queries or not clean_documents:
            return torch.empty((len(clean_queries), len(clean_documents)))

        # Apply instruction to queries only
        query_texts = [f"Instruct: {self.task_instruction}\nPhrase: {q}" for q in clean_queries]
        # document_texts = [f"Instruct: {task_instruction}\nPhrase: {d}" for d in clean_documents]

        # Combine for single batch embedding
        input_texts = query_texts + clean_documents
        # input_texts = clean_queries+ clean_documents
        batch = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Split query and document embeddings
        query_embeddings = embeddings[:len(clean_queries)]
        document_embeddings = embeddings[len(clean_queries):]

        # Cosine similarity
        similarity = query_embeddings @ document_embeddings.T
        return similarity.T

        
    
    def calculate_f1_score(self, tp, fp, fn,pp=None):
        if pp:
            if (2*tp+pp)==0:
                return  0
            elif(2*tp+fp+fn+pp)==0:
                return 0
            else:
                f1_score = (2*tp+pp)/(2*tp+fp+fn+pp)
                return f1_score
        else:
            if (2*tp+pp)==0:
                return  0
            elif(2*tp+fp+fn+pp)==0:
                return 0
            else:
                f1_score = (2*tp)/(2*tp+fp+fn)
                return f1_score

        
    def convert_matrix(self,df_matrix):
        values = df_matrix.values
        columns = df_matrix.columns
        index = df_matrix.index

        # Get row and column indices where value is non-zero
        row_idx, col_idx = np.nonzero(values)

        # Build lists using advanced indexing (fast and order-preserving)
        sources_list = [columns[c] for c in col_idx]
        targets_list = [index[r] for r in row_idx]
        values_list  = [int(values[r, c]) for r, c in zip(row_idx, col_idx)]

        return sources_list, targets_list, values_list
                
    def calculated_node_scores(self,gt_nodes,gen_nodes):
        ScoreCalculator.cache[self.data][self.model_name]['node'] = self.embed_and_score(gt_nodes, gen_nodes).cpu().numpy()
        if 'node' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['node'] = self.embed_and_score(gt_nodes, gen_nodes).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['node'] = self.embed_and_score(gt_nodes, gen_nodes)
        all_scores_node=ScoreCalculator.cache[self.data][self.model_name]['node']
        binary_node_scores = all_scores_node>= self.threshold

        gen_has_tp = np.any(binary_node_scores, axis=1)

        # Apply per-gen filtering to prevent overlap
        matching_edges = matching_edges & gen_has_tp[:, None]


        tp_mask = np.any(matching_edges, axis=0)

        self.TP = np.sum(tp_mask)
        self.FP = np.sum(~(gen_has_tp))
        self.FN = np.sum(~(tp_mask))

        F1 = self.calculate_f1_score(self.TP, self.FP, self.FN)

        self.scores_df = pd.DataFrame(columns=['Model', 'data', 'F1'])
        model_score=pd.DataFrame({'Model': [self.model_name], 'data': [self.data], 'F1': [F1]})
        self.scores_df = pd.concat([self.scores_df,model_score])

        return model_score







    def calculate_scores(self,gt_matrix,gen_matrix):
                
        self.gt_nodes_src,self.gt_nodes_tgt, self.gt_edge_dir=self.convert_matrix(gt_matrix)
        self.gen_nodes_src,  self.gen_nodes_tgt, self.gen_edge_dir=self.convert_matrix(gen_matrix)
        
        # Calculate scores separately for source and target nodes
        # collect cached results if already computed
        if self.data  not in self.cache:
            ScoreCalculator.cache[self.data] = {}

        if self.model not in self.cache[self.data ]:
            ScoreCalculator.cache[self.data][self.model_name] = {}
        if 'src' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gt_nodes_src, self.gen_nodes_src).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['src'] = self.embed_and_score(self.gt_nodes_src, self.gen_nodes_src)

        if 'tgt' not in self.cache[self.data][self.model_name]:
            try:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gt_nodes_tgt, self.gen_nodes_tgt).cpu().numpy()
            except:
                ScoreCalculator.cache[self.data][self.model_name]['tgt'] = self.embed_and_score(self.gt_nodes_tgt, self.gen_nodes_tgt)


        all_scores_src = np.array(ScoreCalculator.cache[self.data][self.model_name]['src'])
        all_scores_tgt = np.array(ScoreCalculator.cache[self.data][self.model_name]['tgt'])

        all_scores_dir = np.zeros((len(self.gen_edge_dir), len(self.gt_edge_dir)))
        for i in range(len(self.gen_edge_dir)):
            for j in range(len(self.gt_edge_dir)):
                if self.gen_edge_dir[i]==self.gt_edge_dir[j]:
                    all_scores_dir[i][j]=True
                else:
                    all_scores_dir[i][j]=False

        all_scores_dir=all_scores_dir.astype(bool)

        binary_matrix_src = all_scores_src >= self.threshold
        binary_matrix_tgt = all_scores_tgt >= self.threshold

        common_mask = binary_matrix_src & binary_matrix_tgt

        matching_edges = common_mask & all_scores_dir
        pp_edges = common_mask & ~all_scores_dir

        gen_has_tp = np.any(matching_edges, axis=1)
        gen_has_pp = np.any(pp_edges, axis=1) & ~gen_has_tp

        # Apply per-gen filtering to prevent overlap
        matching_edges = matching_edges & gen_has_tp[:, None]
        pp_edges = pp_edges & gen_has_pp[:, None]

        tp_mask = np.any(matching_edges, axis=0)
        pp_mask = np.any(pp_edges, axis=0) & ~tp_mask

        self.TP = np.sum(tp_mask)
        self.PP = np.sum(pp_mask)
        self.FP = np.sum(~(gen_has_tp | gen_has_pp))
        self.FN = np.sum(~(tp_mask | pp_mask))

        TP=TP*self.tp_scale
        PP=PP*self.pp_scale
        
        F1 = self.calculate_f1_score(TP, self.FP, self.FN, PP)
        self.scores_df = pd.DataFrame(columns=['Model', 'data', 'F1'])
        model_score=pd.DataFrame({'Model': [self.model_name], 'data': [self.data], 'F1': [F1]})
        self.scores_df = pd.concat([self.scores_df,model_score])

        return model_score
    


# def main():
#     # scorer = ScoreCalculator(0.5, model_name, folder, 0.4, 0.1)
#     # scorer.calculate_scores(gt_matrix, df_matrix)
#     # scorer.scores_df ****output df contains score****

# if __name__ == "__main__":
#   main()