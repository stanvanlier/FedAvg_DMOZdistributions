import pkg_resources

ROOT_DIR = pkg_resources.resource_filename("src", "..")

def get_dmoz_file(file_name, source_path="https://curlz.org/dmoz_rdf/", save_path="data/external", verbose=1):
    import requests
    import gzip
    import os

    if verbose > 0:
        print(file_name, end=": downloading... ")

    resp = requests.get(os.path.join(source_path, file_name + ".gz"), verify=False)

    if verbose > 0:
        print("decompressing ...", end="")

    os.makedirs(os.path.join(ROOT_DIR, save_path), exist_ok=True)
    with open(os.path.join(ROOT_DIR, save_path, file_name), "wb") as f:
        f.write(gzip.decompress(resp.content))

    if verbose > 0:
        print("saved at", os.path.join(ROOT_DIR, save_path, file_name))


def make_content_tsv(which="main", verbose=1, force_download=False):
    import os

    if which == "main":
        file = "content"
    else:
        file = "{}-content".format(which)
    
    rdf_file = file + ".rdf.u8"
    tsv_file = file + ".tsv" 
    data_external = "data/external"
    data_interim = "data/interim"

    if not os.path.exists(os.path.join(ROOT_DIR, data_external, rdf_file)) or force_download:
        print("RDF file not found, trying to download.")
        get_dmoz_file(rdf_file, save_path=data_external, verbose=verbose)

    def line2dict(line):
        key = line[line.index('<')+1:line.index('>')]
        start = line.index("<"+key+">") + len(key)+2
        end = line.index("</"+key+">")
        return {key:line[start:end]}

    cols = ['topic', 'd:Title', 'd:Description', 'about', 'priority', 'mediadate', 'type']
    header = [s[s.find(":")+1:].lower() for s in cols]

    os.makedirs(os.path.join(ROOT_DIR, data_interim), exist_ok=True)
    with open(os.path.join(ROOT_DIR, data_interim, tsv_file), "w") as o:
        if verbose > 0:
            lines = 0
            print(tsv_file, end=" << ")

        o.write("\t".join(header)+"\n")

        with open(os.path.join(ROOT_DIR, data_external, rdf_file), "r") as f:
            for l in f:
                l = l.strip()
                if l.startswith("<ExternalPage "):        # start of page
                    page_dict = {"about":l[l.index('"')+1:l.rindex('"')]}
                    l = f.readline().strip()

                    while l != "</ExternalPage>":         # continue until all fields are parsed
                        page_dict.update(line2dict(l))
                        l = f.readline().strip()
                        
                    topic = page_dict.get("topic")
                    # if not topic.startswith(exclude_cats): # if topic is not excluded
                    o.write("\t".join(map(lambda x: page_dict.get(x, ""), cols))+"\n")
                    if verbose > 0:
                        lines += 1
                        if lines % 500000 == 0:
                            print(lines, end="... ")
    if verbose > 0:
        print(lines, "lines")

    return os.path.join(ROOT_DIR, data_interim, tsv_file)


def make_structure_tsv(which="main", verbose=1, force_download=False):
    import os

    if which == "main":
        file = "structure"
    else:
        file = "{}-structure".format(which)
    
    rdf_file = file + ".rdf.u8"
    tsv_file = file + ".tsv" 
    data_external = "data/external"
    data_interim = "data/interim"

    if not os.path.exists(os.path.join(ROOT_DIR, data_external, rdf_file)) or force_download:
        print("RDF file not found, trying to download.")
        get_dmoz_file(rdf_file, save_path=data_external, verbose=verbose)
    
    header = ['topic','resource','tag']

    os.makedirs(os.path.join(ROOT_DIR, data_interim), exist_ok=True)
    with open(os.path.join(ROOT_DIR, data_interim, tsv_file), 'w') as w:
        if verbose > 0:
            lines = 0
            print(tsv_file, end=" << ")

        w.write("\t".join(header)+"\n")

        with open(os.path.join(ROOT_DIR, data_external, rdf_file), "r") as r:
            for l in r:
                l = l.strip()
                if l.startswith("<Topic "):                             #start parsing relations of whole topic
                    topic = l[l.index('r:id="')+6:l.rindex('"')]
                    if len(topic) == 0:
                        continue
                    l = r.readline().strip()
                    while l != "</Topic>":                              #continue parsing rows until topic ends
                        tag = l[1:l.find(' ')]
                        if tag.isalnum() and tag.islower():
                            resource = l[l.index('"')+1:l.rindex('"')].strip()
                            w.write("\t".join([topic, resource, tag])+"\n")
                            if verbose > 0:
                                lines +=1
                                if lines % 500000 == 0:
                                    print(lines, end="... ")
                        l = r.readline().strip()
    if verbose > 0:
        print(lines, "lines")

    return os.path.join(ROOT_DIR, data_interim, tsv_file)


if __name__ == "__main__":
    content_file = make_content_tsv(which="main")
    structure_file = make_structure_tsv(which="main")

    print(content_file, "and", structure_file, "created.")


