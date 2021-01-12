import json


def id_finder(list_var):
    for list_item in list_var:
        for k, v in list_item.items():
            if k == "id":
                yield v


if __name__ == '__main__':
    # input files to process
    names = ['famcov_new.json', 'property_coalesce.json']

    # init the set of IDs
    id_set: set = set()

    # for each json file
    for name in names:
        # open the json file
        json_file = open(name, 'r+')

        # load the json
        json_data = json.loads(json_file.read())

        # for each id in the KG nodes
        for node_id in id_finder(json_data['knowledge_graph']['nodes']):
            # add in the id to the set
            id_set.add(node_id)

    # sort the set
    ids = sorted(id_set)

    # print(ids)

    # init the set that holds the curie prefixes
    prefixes = set()

    # get a set of unique curie prefixes
    for x in ids:
        prefixes.add(x.split(':')[0].lower())

    # print(prefixes)

    # for each curie type
    for item in prefixes:
        print(f'Working: {item}')

        # open the output file
        with open(f'omnicorp_{item}.csv', 'w') as output_file:
            output_file.write('curie, pubmedid\n')

            # open the input file
            with open(f'D:/Work/Robokop/Data_services/omnicorp_data/{item}', "r") as in_file:
                # read in the records
                for line in in_file:
                    # get the line into the two parts
                    key, value = line.rstrip().split('\t')

                    # if this is what we are looking for and is a proper PMID
                    if value in id_set and key.startswith('PMID:'):
                        # write out the data to the CSV file
                        output_file.write(f"{value},{key.split(':')[1]}\n")
