import sys
import extraction
import query

if len(sys.argv) == 3 and sys.argv[1] == "create_index":
    extraction.build_index(sys.argv[2])
elif len(sys.argv) == 4 and sys.argv[1] == "query":
    results = query.query(sys.argv[2], sys.argv[3])
    output_file = open("ranked_query_docs.txt", "w")
    for tup in results:
        output_file.write(str(tup[0]) + "\n")
    output_file.close()
else:
    pass
