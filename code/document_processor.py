"""
文档处理模块
支持多种文档格式的加载和处理
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    PDFPlumberLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载单个文档"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.md':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.pdf':
                loader = PDFPlumberLoader(file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_ext}")
                return []
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个片段")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            return []
    
    def load_documents_from_directory(self, directory: str, 
                                    supported_extensions: List[str] = None) -> List[Document]:
        """从目录加载所有支持的文档"""
        if supported_extensions is None:
            supported_extensions = ['.txt', '.md', '.pdf', '.docx', '.csv']
        
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"目录不存在: {directory}")
            return documents
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_document(str(file_path))
                documents.extend(docs)
        
        logger.info(f"从目录 {directory} 加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        if not documents:
            return []
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档分割完成: {len(documents)} -> {len(split_docs)} 个片段")
            return split_docs
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            return []
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """处理多个文档文件"""
        all_documents = []
        
        for file_path in file_paths:
            docs = self.load_document(file_path)
            all_documents.extend(docs)
        
        return self.split_documents(all_documents)
    
    def create_sample_document(self, file_path: str) -> bool:
        """创建示例文档"""
        try:
            sample_content = """
# RAG系统学习指南

## 什么是RAG？

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术架构。

### 核心组件

1. **检索器（Retriever）**
   - 负责从知识库中找到相关信息
   - 通常使用向量相似度搜索
   - 支持多种检索策略

2. **生成器（Generator）**
   - 基于检索到的信息生成回答
   - 通常使用大语言模型
   - 能够整合多个信息源

### 技术优势

- **知识更新**：可以实时更新知识库
- **可解释性**：能够追踪信息来源
- **准确性**：减少模型幻觉问题
- **灵活性**：支持多种应用场景

### 应用场景

1. **问答系统**：企业内部知识问答
2. **文档助手**：智能文档分析
3. **客服机器人**：基于知识库的智能客服
4. **研究助手**：学术论文检索与分析

### 实现步骤

1. 文档预处理和分块
2. 向量化处理
3. 构建向量数据库
4. 实现检索机制
5. 集成生成模型
6. 系统优化和评估

这是一个完整的RAG系统实现示例。
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            
            logger.info(f"创建示例文档: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建示例文档失败: {str(e)}")
            return False 