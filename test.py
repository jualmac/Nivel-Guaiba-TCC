from source_database.db_handler_duck import DBConnection

db = DBConnection()

query = 'SELECT * FROM "main"."aaa"'
results = db.run(query=query)
print (results)