# haoo
hi!
happy birthday

tool 
def query_production_database(question: str) -> str:
    """
    當使用者問到有關查詢資料庫之訊息，如TankID、BatchID 等等需要去資料庫找資料時請使用此道具
    特別注意，最終的查詢結果必須是好看，條列式的資料。如:Crate_Id: XXXX Batch Id: xxxx
    如果資料最終查詢結果不只一筆，你需要說總共有幾筆資料，以及所有他們的訊息，就算他們的數值重複也要說明，因為他們可能來自不同的資料。
    最後輸出的結果請幫我搭配表格名稱。請特別注意回覆時請使用正確、原始的表格名稱，請勿自己翻譯。
    """
    print(f"--- [CrateTracker Tool] 正在執行 資料庫查詢工具，問題: {question} ---")
    try: 
        llm = load_llm("gpt-4o")
        embeddings = load_embeddings()
        
        # 1. 載入業務說明 (RAG)
        loader = TextLoader(r'C:\Users\wangl100\programing\orion_agent\agents\crate_tracker_agent\test.txt', encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        vector_db = FAISS.from_documents(docs, embeddings)
        retriever = vector_db.as_retriever()
        print("業務說明載入完成")
        
        # 2. 動態獲取資料庫結構
        db = SQLDatabase.from_uri()
        
        schema_query = """
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA NOT IN ('sys', 'information_schema')
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        
        current_schema = db.run(schema_query)
        print("資料庫結構獲取完成")
        
        # 3. 整合模板
        sql_template = """
        你是一位專業的玻璃製造業 SQL 查詢專家，根據以下資訊產生合法、乾淨的 SQL 查詢語句。

        【業務說明和範例】：
        {business_context}

        【當前資料庫結構】：
        {db_structure}

        查詢規則：
        1. 禁止使用 INSERT、DELETE、UPDATE
        2. 只回傳純粹的 SQL 語法，不要使用 markdown 語法
        3. 使用 Microsoft SQL Server 語法
        4. 若使用者沒有特別說明，請查找所有結果，而非使用 TOP 1
        5. 欄位名稱必須與資料庫結構完全一致
        6. 僅根據現有資料回覆，不得捏造資料

        使用者問題：{user_query}

        SQL查詢語法：
        """
        
        prompt = ChatPromptTemplate.from_template(sql_template)
        
        # 4. 建立查詢鏈
        sql_chain = (
            {
                "business_context": retriever,              # RAG 提供業務說明
                "db_structure": lambda x: current_schema,   # 動態結構
                "user_query": RunnablePassthrough()         # 使用者問題
            }
            | prompt
            | llm 
            | StrOutputParser()
        )
        
        # 5. 執行查詢
        sql_query = sql_chain.invoke(question)
        print(f"生成的 SQL: {sql_query}")
        
        query_results = db.run(sql_query)
        print(f"查詢結果: {query_results}")
        
        return {"output": str(query_results)}
        
    except Exception as e:
        return f"執行資料庫查詢時發生錯誤: {e}"
