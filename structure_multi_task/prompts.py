# prompts.py
"""
Prompt管理模块 - 集中管理所有的prompt模板
"""

class PromptManager:
    """Prompt管理器"""
    @staticmethod
    def get_relation_extraction_prompt(data_source: str, text: str, entity1: str=None, entity2: str=None) -> str:
        """获取关系抽取的prompt - 改进版本"""
        if data_source == 'chemport':
            prompt = f"""[Task: RE][Domain: bigbio/chemprot]. You are an expert in biomedical relation extraction. 
                Given a text and two entities, identify the relationship between the entities based on the predefined relation types.
                Relation types:- CPR:3: UPREGULATOR- CPR:4: DOWNREGULATOR- CPR:5: AGONIST- CPR:6: ANTAGONIST- CPR:9: SUBSTRATE- CPR:10: NO. 
                What is the relationship between {entity1} acid and {entity2} in the following text. Respond with: 'CPR:3', 'CPR:4','CPR:5','CPR:6''CPR:9'.
                {text}
                """
            
            return prompt
        
        elif data_source == 'ddi':
            prompt = f"""[Task: RE][Domain: bigbio/ddi_corpus]. You are an expert in biomedical relation extraction. 
            Given a sentence containing two marked entities (@DRUG$ and @DRUG$), 
            determine whether there is a meaningful biomedical relationship between them. 
            Respond with: '0', 'DDI-mechanism', 'DDI-effect', 'DDI-advise', 'DDI-int'. Input: {text}
            """
            
            return prompt
        
        elif data_source == 'gad':
            prompt = f"""[Task: RE][Domain: bigbio/gad]. You are an expert in biomedical relation extraction.
            Given a sentence containing two marked entities (@GENE$ and @DISEASE$), determine whether there is a meaningful biomedical relationship between them. Respond with either 1 or 0.
            Input: {text}
            """
            
            return prompt

    @staticmethod
    def get_ner_prompt(data_source: str, text: str, tokens: str, info: str) -> str:
        """获取关系抽取的prompt - 改进版本"""
        if data_source == 'bc2gm':
            prompt = f"""[Task: NER][Domain: spyysalo/bc2gm_corpus] You are a biomedical named entity recognition (NER) expert. 
            Please extract and label all predefined biomedical entities using the BIO format. 
            Supported entity labels list (BIO format): ['O', 'B-DISEASE', 'I-DISEASE']. 
            Input: {text}. Tokens: {tokens}
            """

            return prompt
        elif data_source == 'jnlpba':
            prompt = f"""[Task: NER][Domain: jnlpba/jnlpba] You are a biomedical named entity recognition (NER) expert. 
            Please extract and label all predefined biomedical entities using the BIO format. 
            Supported entity labels list (BIO format): ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell_line', 'I-cell_line', 'B-cell_type', 'I-cell_type', 'B-protein', 'I-protein']. 
            Input: {text} Tokens: {tokens}
            """

            return prompt
        elif data_source == 'ncbi':
            prompt = f"""[Task: NER][Domain: NCBI-disease].You are a biomedical named entity recognition (NER) expert. 
            Please extract and label all predefined biomedical entities using the BIO format. Supported entity labels list (BIO format): ['O', 'B', 'I']. 
            Input: {text}. Tokens: {tokens}
            """

            return prompt

    @staticmethod
    def get_qa_prompt(data_source, question: str, context: str) -> str:
        """获取关系抽取的prompt - 改进版本"""
        if data_source == 'pubmedqa':
            prompt = f"""[Task: QA][Domain: qiaojin/PubMedQA]. 
            You are an expert in biomedical question answering. 
            Based on the following context and question, response with: 'yes', 'no', 'maybe'.
            Input: Question: {question}\n\n Context: {context}.
            """

            return prompt
        
        elif data_source == 'bioasq':
            prompt = f"""[Task: QA][Domain: kroshan/BioASQ]. 
            You are an expert in biomedical question answering. 
            Based on the following context and question, response with: 'yes' or 'no'.
            Input: Question: {question}. Context: {context}
            """

            return prompt

        elif data_source == 'medqa':
            prompt = f"""[Task: QA][Domain: GBaker/MedQA-USMLE-4-options]. 
            You are an expert in biomedical question answering. Given a question and four multiple choice options (A, B, C, D), 
            select the most appropriate answer based on your biomedical knowledge.Respond with only the letter (A, B, C, or D) corresponding to your chosen answer.
            Input: Question: {question}, Options: {context}.
            """

            return prompt