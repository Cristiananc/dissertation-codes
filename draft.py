for i, path in enumerate([[0], [1,2,3]]):
            # Assuming format: path = [leaf/start_node, ..., root]
            if path and path[0] == 1:
                print(path)