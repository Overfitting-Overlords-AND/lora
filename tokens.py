import transformers
import config

tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

if not config.BASE_MODEL:
    tokenizer.add_special_tokens({'additional_special_tokens':[
    "CREATE",
    "TABLE",
    "INTEGER",
    "SELECT",
    "COUNT(*)",
    "COUNT",
    "FROM",
    "WHERE",
    "VARCHAR",
    "ORDER",
    "MAX",
    "MIN",
    "AVG",
    "BETWEEN",
    "DISTINCT",
    "JOIN",
    "AS",
    "ON",
    "GROUP",
    "BY",
    "HAVING",
    "LIMIT",
    "ASC",
    "DESC",
    "LIKE",
    "INTERSECT",
    "EXCEPT"                              
    ]})
