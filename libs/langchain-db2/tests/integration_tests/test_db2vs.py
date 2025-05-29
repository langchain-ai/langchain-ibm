"""Test Db2 AI Vector Search functionality."""

# import required modules
import sys
import threading

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_db2.db2vs import (
    DB2VS,
    _create_table,
    _table_exists,
    drop_table,
)
from langchain_community.vectorstores.utils import DistanceStrategy

database = ""
username = ""
password = ""

############################
####### table_exists #######
############################
def test_table_exists_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)

    # 1. Create a Table
    _create_table(connection, "TB1", 8148)

    # 2. Existing Table
    # expectation:true    
    assert _table_exists(connection, "TB1")

    # 3. Non-Existing Table
    # expectation:false
    assert _table_exists(connection, "TableNonExist") == False

    # 4. Invalid Table Name
    # Expectation: SQL0104N 
    try:
        _table_exists(connection, "123")
    except Exception:
        pass

    # 5. Empty String
    # Expectation: SQL0104N
    try:
        _table_exists(connection, "")
    except Exception:
        pass

    # 6. Special Character
    # Expectation: SQL0007N
    try:
        _table_exists(connection, "!!4")
    except Exception:
        pass

    # 7. Table name length > 128
    # Expectation: SQL0107N The name is too long.  The maximum length is "128".
    try:
        _table_exists(connection, "x" * 129)
    except Exception:
        pass

    # 8. Toggle Upper/Lower Case (like TaBlE)
    # Expectation:True
    assert _table_exists(connection, "Tb1")
    drop_table(connection, "TB1")

    # 9. Table_Name→ "表格"
    # Expectation:True
    _create_table(connection, '"表格"', 545)
    assert _table_exists(connection, '"表格"')
    drop_table(connection, '"表格"')

    connection.commit()


############################
####### create_table #######
############################
def test_create_table_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation: table is created
    _create_table(connection, "HELLO", 100)

    # 2. Existing table name - HELLO
    #    Dimension - 110
    # Expectation: Log message table already exists
    _create_table(connection, "HELLO", 110)
    drop_table(connection, "HELLO")

    # 3. New Table - 123
    #    Dimension - 100
    # Expectation: SQL0104N  invalid table name
    try:
        _create_table(connection, "123", 100)
        drop_table(connection, "123")
    except Exception:
        pass

    # 4. New Table - Hello123
    #    Dimension - 8148
    # Expectation: table is created
    _create_table(connection, "Hello123", 8148)
    drop_table(connection, "Hello123")

    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation: SQL0604N  VECTOR column exceed the supported 
    # dimension length.
    try:
        _create_table(connection, "T1", 65536)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 6. New Table - T1
    #    Dimension - 0
    # Expectation: SQL0604N  VECTOR column unsupported dimension length 0.
    try:
        _create_table(connection, "T1", 0)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 7. New Table - T1
    #    Dimension - -1
    # Expectation: SQL0104N  An unexpected token "-" was found
    try:
        _create_table(connection, "T1", -1)
        drop_table(connection, "T1")
    except Exception:
        pass

    # 8. New Table - T2
    #     Dimension - '1000'
    # Expectation: table is created
    _create_table(connection, "T2", int("1000"))
    drop_table(connection, "T2")

    # 9. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation: table is created
    val = 100
    _create_table(connection, "T3", val)
    drop_table(connection, "T3")

    # 10.
    # Expectation: SQL0104N  An unexpected token
    val2 = """H
    ello"""
    try:
        _create_table(connection, val2, 545)
        drop_table(connection, val2)
    except Exception:
        pass

    # 11. New Table - 表格
    #     Dimension - 545
    # Expectation: table is created
    _create_table(connection, '"表格"', 545)
    drop_table(connection, '"表格"')

    # 12. <schema_name.table_name>
    # Expectation: table with schema is created
    _create_table(connection, "U1.TB4", 128)
    drop_table(connection, "U1.TB4")

    # 13.
    # Expectation: table is created
    _create_table(connection, '"T5"', 128)
    drop_table(connection, '"T5"')

    # 14. Toggle Case
    # Expectation: table is created
    _create_table(connection, "TaBlE", 128)
    drop_table(connection, "TaBlE")

    # 15. table_name as empty_string
    # Expectation: SQL0104N  An unexpected token
    try:
        _create_table(connection, "", 128)
        drop_table(connection, "")
        _create_table(connection, '""', 128)
        drop_table(connection, '""')
    except Exception:
        pass

    # 16. Arithmetic Operations in dimension parameter
    # Expectation: table is created
    n = 1
    _create_table(connection, "T10", n + 500)
    drop_table(connection, "T10")

    # 17. String Operations in table_name parameter
    # Expectation: table is created
    _create_table(connection, "YaSh".replace("aS", "ok"), 500)
    drop_table(connection, "YaSh".replace("aS", "ok"))

    connection.commit()

##################################
####### add_texts ################
##################################
def test_add_texts_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)
    
    # 1. Add 2 records to table
    # Expectation: Successful
    texts = ["David", "Vectoria"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = DB2VS(connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts, metadata)
    drop_table(connection, "TB1")

    # 2-1. Add 2 records to table with metadata but no id inside it 
    # Expectation: Successful, new ID will be generated
    metadataNoID = [
        {"link": "Document Example Test 1"},
        {"link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = DB2VS(connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts, metadataNoID)
    drop_table(connection, "TB2")

    # 2-2. Add 3 records to table with metadata but partial metadata has id
    # Expectation: Successful, new ID will be generated for the missing ones
    texts1 = ["David", "Vectoria", "John"]
    metadataPartialID = [
        {"id": "100", "link": "Document Example Test 1"},
        {"link": "Document Example Test 2"},
        {"link": "Document Example Test 3"},
    ]
    vs_obj = DB2VS(connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts1, metadataPartialID)
    drop_table(connection, "TB2")

    # 3. Add record but neither metadata nor ids are there
    # Expectation: Successful, new ID will be generated
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = DB2VS(connection, model, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts2 = ["Sam", "John"]
    vs_obj.add_texts(texts2)
    drop_table(connection, "TB3")

    # 4. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = DB2VS(connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids4 = ["114", "124"]
    vs_obj.add_texts(texts2, ids=ids4)
    drop_table(connection, "TB4")

    vs_obj = DB2VS(connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids5 = ["", "134"]
    vs_obj.add_texts(texts2, ids=ids5)
    drop_table(connection, "TB5")

    vs_obj = DB2VS(connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids6 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    vs_obj.add_texts(texts2, ids=ids6)
    drop_table(connection, "TB6")

    vs_obj = DB2VS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids7 = ['"Good afternoon"', '"India"']
    vs_obj.add_texts(texts2, ids=ids7)
    drop_table(connection, "TB7")

    # 5. Add record with ids option but the id are duplicated
    # Expectations: SQL0803N having duplicate values for the index key
    try:
        vs_obj = DB2VS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
        ids8 = ["118", "118"]
        vs_obj.add_texts(texts2, ids=ids8)
        drop_table(connection, "TB8")
    except Exception:
        pass

    # 6. Add records with both ids and metadatas
    # Expectation: Successful, the ID will be generated based on ids
    vs_obj = DB2VS(connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts3 = ["Sam 6", "John 6"]
    ids9 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    vs_obj.add_texts(texts3, metadata, ids=ids9)
    drop_table(connection, "TB9")

    # This one may run slow before using executemany() <<<<<<<<<<
    # 7. Add 10000 records
    # Expectation:Successful
    vs_obj = DB2VS(connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts4 = ["Sam{0}".format(i) for i in range(1, 10000)]
    ids10 = ["Hello{0}".format(i) for i in range(1, 10000)]
    vs_obj.add_texts(texts4, ids=ids10)
    drop_table(connection, "TB10")

    # 8. Add 2 different record concurrently
    # Expectation:Successful
    def add(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = DB2VS(
            connection, model, "TB11", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts5 = [val]
        ids11 = texts5
        vs_obj.add_texts(texts5, ids=ids11)

    thread_1 = threading.Thread(target=add, args=("Sam",))
    thread_2 = threading.Thread(target=add, args=("John",))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    drop_table(connection, "TB11")

    # 9. Add 2 same record concurrently
    # Expectation: Successful, For one of the insert,get primary key violation error
    def add1(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = DB2VS(
            connection, model, "TB12", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts = [val]
        ids12 = texts
        vs_obj.add_texts(texts, ids=ids12)

    try:
        thread_1 = threading.Thread(target=add1, args=("Sam",))
        thread_2 = threading.Thread(target=add1, args=("Sam",))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()
    except Exception:
        pass
    drop_table(connection, "TB12")

    connection.commit()


##################################
####### embed_documents(texts) ####
##################################
def test_embed_documents_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)
    # 1. Embed String Example-'Sam'
    # Expectation: Successful. Vector Printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = DB2VS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    print(vs_obj._embed_documents(["Sam",]))

    # 2. Embed List of string
    # Expectation: Successful. Vector Printed
    print(vs_obj._embed_documents(["hello", "yash"]))
    drop_table(connection, "TB7")

    connection.commit()

##################################
####### embed_query(text) ########
##################################
def test_embed_query_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)

    # 1. Embed String
    # Expectation: Successful. Vector printed
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = DB2VS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    print(vs_obj._embed_query("Sam"))

    # 3. Embed Empty string
    # Expectation: Successful. Vector printed
    print(vs_obj._embed_query(""))
    drop_table(connection, "TB8")

    connection.commit()

##################################
####### perform_search ###########
##################################
def test_perform_search_test() -> None:
    try:
        import ibm_db_dbi
    except ImportError:
        return

    try:
        connection = ibm_db_dbi.connect(database, username, password)
    except Exception:
        sys.exit(1)
    
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs_1 = DB2VS(connection, model1, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_2 = DB2VS(connection, model1, "TB11", DistanceStrategy.DOT_PRODUCT)
    vs_3 = DB2VS(connection, model1, "TB12", DistanceStrategy.COSINE)

    # vector store lists:
    vs_list = [vs_1, vs_2, vs_3]

    for i, vs in enumerate(vs_list, start=1):
        # insert data
        texts = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
        metadatas = [
            {"id": "hello"},
            {"id": "105"},
            {"id": "106"},
            {"id": "yash"},
            {"id": "108"},
        ]

        vs.add_texts(texts, metadatas)

        # perform search
        query = "YashB"

        filter = {"id": ["106", "108", "yash"]}

        # similarity_search without filter
        vs.similarity_search(query, 2)

        # similarity_search with filter
        vs.similarity_search(query, 2, filter=filter)

        # Similarity search with relevance score
        vs.similarity_search_with_score(query, 2)

        # Similarity search with relevance score with filter
        vs.similarity_search_with_score(query, 2, filter=filter)

        # Max marginal relevance search
        vs.max_marginal_relevance_search(query, 2, fetch_k=20, lambda_mult=0.5)

        # Max marginal relevance search with filter
        vs.max_marginal_relevance_search(
            query, 2, fetch_k=20, lambda_mult=0.5, filter=filter
        )

    drop_table(connection, "TB10")
    drop_table(connection, "TB11")
    drop_table(connection, "TB12")

    connection.commit()
