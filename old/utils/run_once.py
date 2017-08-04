# coding:utf-8

def get_file_list(dirpath):
    from os import listdir
    path = dirpath
    file_list = ''
    files = listdir(path)
    for file in sorted(files):
        file_list = file_list + '\n' + file
    return file_list

kdd_cited = get_file_list('./data/KDD/citedcontexts')
kdd_citing = get_file_list('./data/KDD/citingcontexts')
www_cited = get_file_list('./data/WWW/citedcontexts')
www_citing = get_file_list('./data/WWW/citingcontexts')

def write_file(context, file):
    with open(file,'w') as f:
        f.write(context)
write_file(kdd_cited,'./data/kdd_cited')
write_file(kdd_citing,'./data/kdd_citing')
write_file(www_cited,'./data/www_cited')
write_file(www_citing,'./data/www_citing')