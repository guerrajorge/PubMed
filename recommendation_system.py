from Bio import Entrez
import csv
import logging
from utils.logger import logger_initialization
from utils.parse import parse
import argparse
import pandas as pd
import pandas.io.formats.excel
from utils.topic import topic_modeling


def obtain_descriptions():

    # get the description, related to the MESH, in the 2017MeshTree.csv File
    mesh_tree_file_object = open(r'C:\Users\GUERRAMARJ\PycharmProjects\Pubmed\template\2017MeshTree.csv')
    file_reader = csv.reader(mesh_tree_file_object, delimiter=',')
    mesh_description_dict = dict()

    logging.getLogger('regular').info('processing each record and obtaining relevant information')
    for line in file_reader:
        # split_line[0] = Number, split_line[1] = Description and split_line[2] = MESH
        mesh_description_dict[line[2]] = line[1]
    mesh_tree_file_object.close()

    return mesh_description_dict


def assign_roles(author_list):
    """
    assign the chief author, ordinary author or principal investigator role to each author
    :param author_list: a list of all the authors in the paper
    :return: role_list: the authors' respective roles
    """

    role_list = list()

    for author_index in range(len(author_list)):
        # Assign the author's rle
        # if less than 2 authors then they are considered "Chief Authors"
        if author_index <= 1:
            role_list.append('CA')
        # If a person is after the first two authors and it'snt the last author its considered
        # "Ordinary Author"
        elif author_index > 1 and author_index != len(author_list) - 1:
            role_list.append('OA')
        # else "Principal Investigator)
        elif author_index == len(author_list) - 1:
            role_list.append('PI')

    return role_list


def assign_organization(affiliation_list):
    """
    check and assign whether the authors belong to the CHOP or PENN organization
    :param affiliation_list: a list of all the affiliations of the authors
    :return: chop_list, penn_list: lists with whether the author belong to the CHOP or PENN organization
    """
    # initialize CHOP and PENN authors' organization to None = 0
    chop_list = [0] * len(affiliation_list)
    penn_list = [0] * len(affiliation_list)

    for affiliation_index, affiliation in enumerate(affiliation_list):

        sub_affiliation = affiliation.split(';')

        for sa in sub_affiliation:
            # Assign the author organization
            if 'children' in sa.lower():
                chop_list[affiliation_index] = 1
                break
            elif 'perelman' in sa.lower() or 'school of medicine' in sa.lower() or \
                 'pennsylvania' in affiliation.lower():
                penn_list[affiliation_index] = 1
                break

    return chop_list, penn_list


def convert_mesh_description(mesh_term_description_dict, mesh_term):
    """
    convert the mesh_term found for that paper to the mesh description from the 2017MeshTree.csv
    :param mesh_term_description_dict: a dictionary where key=mesh term, value = mesh description
    :param mesh_term: the mesh term(s) of the paper
    :return: term, description: the term and its description
    """
    # fetch the description from the description obtain from the 2017MeshTree file
    if len(mesh_term) > 1:

        # because there are mesh_term that are not part of the 2017MeshTree, we have to loop through all
        # of the mesh_term until one works i.e. the first one found in the 2017MeshTree
        for mesh in mesh_term:

            try:
                term = mesh

                # cleaning string
                if '/' in term:
                    term = term.split('/')[0]
                if '*' in term:
                    term = term.replace('*', '')

                logging.getLogger('regular').debug('term = {0}'.format(term))

                description = mesh_term_description_dict[term]

            except KeyError:
                logging.getLogger('regular').debug('not found term = {0}'.format(term))
                continue
    else:

        if mesh_term not in mesh_term_description_dict.keys():
            raise KeyError('mesh term = {0} not found'.format(mesh_term))
        else:
            description = mesh_term_description_dict[mesh_term]

    return description, term


def print_str(*args):
    n_string = ''
    for element in args:
        n_string += '"{0}",'.format(element)
    n_string = n_string[:-1]
    n_string += '\n'
    return n_string


def main():
    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('line.regular.time.line').info('Running Recommendation System script')

    # import data from file
    logging.getLogger('regular').info('reading data from file')

    # Entrez (http://www.ncbi.nlm.nih.gov/Entrez) is a data retrieval system that provides users access to NCBI’s
    # databases such as PubMed, GenBank, GEO, and many others
    # Use the mandatory email parameter so the NCBI can contact you if there is a proble
    Entrez.email = "guerramarj@email.chop.edu"     # Always tell NCBI who you are
    # logging.getLogger('regular').info('searching pubmed for the CHOP and UPENN authors')
    # handle = Entrez.esearch(db="pubmed", retmax=50000, idtype="esearch", mindate="2014/01/01", maxdate="2017/05/01",
    #                         term="Perelman School of Medicine[Affiliation] OR Children's Hospital of "
    #                              "Philadelphia[Affiliation] OR University of Pennsylvania School of "
    #                              "Medicine[Affiliation] OR School of Medicine University of Pennsylvania[Affiliation]",
    #                         usehistory="y")
    # search_results = Entrez.read(handle)
    # handle.close()
    # # obtaining the list of relevant PMIDs
    # id_list = search_results["IdList"]
    #
    # # get all the record based on the PMIDs
    # logging.getLogger('regular').info('getting relevant authors\' records based on PMIDs')
    # fetch_records_handle = Entrez.efetch(db="pubmed", id=id_list, retmode="text", rettype="medline")
    # # need to read all the data from the handle and store in a file because if we just read line by line from the
    # # generator and the internet connection is not strong, then we run into http errors:
    # # http.client.IncompleteRead: IncompleteRead(0 bytes read)
    # logging.getLogger('regular').info('storing authors\' records on local file')
    # with open("results.xml", "w") as out_handle:
    #     out_handle.write(fetch_records_handle.read())
    # # the results are now in the results.xml file and the original handle has had all of its data extracted
    # # (so we close it)
    # fetch_records_handle.close()

    logging.getLogger('regular').info('reading result files')
    records_handle = open("results.xml")
    fetch_records = parse(handle=records_handle)

    # initializing variables
    mesh_description_dict = obtain_descriptions()

    # contains all the metadata elements on the author level: Pubmed unique Identifier number(PMID), AuthorID (as a
    # combination of the author’s last name, first name, and initials), institution: chop=0, Penn=1, Role: Chief Author
    # (CA) Ordinary Author (OA) or Principal Author (PA) and the author's affiliation
    author_record_df = pd.DataFrame(columns=['PMID', 'AuthorID', 'Author CHOP', 'Author PENN', 'ROLE', 'Affiliation'])
    # contains all the metadata elements on the paper level: Pubmed unique Identifier number(PMID), Title, Abstract,
    # Year, Month, AuthorList, SubjectList, date
    paper_record_df = pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Year', 'Month', 'Author List', 'Subject List',
                                            'Date'])
    # contains all the metadata of the medical information: Pubmed unique Identifier number(PMID), Primary Medical
    # Subject Header (MESH) and the description ID
    medical_record_df = pd.DataFrame(columns=['PMID', 'MESH', 'Description'])

    title_list = list()
    abstract_list = list()

    # get the relevant information for each record
    for record_index, record in enumerate(fetch_records):

        logging.getLogger('regular').debug('record index = {0}'.format(record_index))

        try:
            pmid = record.get('PMID')
            title = record.get('TI')
            abstract = record.get('AB')
            authors = record.get('FAU')
            affiliations = record.get('AD')
            publication_type = record.get('PT')
            mesh_term = record.get('MH')
            date_created = record.get('EDAT')
            year, month = date_created.split('/')[:2]
            date = year + '/' + month

            logging.getLogger('regular').debug('pmid = {0}'.format(pmid))
            logging.getLogger('regular').debug('title = {0}'.format(title))
            logging.getLogger('regular').debug('abstract = {0}'.format(abstract))
            logging.getLogger('regular').debug('authors = {0}'.format(authors))
            logging.getLogger('regular').debug('affiliations = {0}'.format(affiliations))
            logging.getLogger('regular').debug('publication type = {0}'.format(publication_type))
            logging.getLogger('regular').debug('mesh term = {0}'.format(mesh_term))
            logging.getLogger('regular').debug('data created = {0}'.format(date_created))

            # assign the chief author, ordinary author or principal investigator role to each author
            roles = assign_roles(authors)
            # check and assign whether the authors belong to the CHOP or PENN organization
            chop_organization, penn_organization = assign_organization(affiliations)

            mesh_description = ''
            if mesh_term is None:
                mesh_term = ''
            else:
                term, mesh_description = convert_mesh_description(mesh_description_dict, mesh_term)
                mesh_term = ';'.join(mesh_term)

            # output information
            if mesh_description:
                row = pd.DataFrame([[pmid, term, mesh_description]], columns=['PMID', 'Mesh', 'Description'])
                medical_record_df = medical_record_df.append(row, ignore_index=True)

            for author_index, organizations in enumerate(zip(chop_organization, penn_organization)):
                if 1 in organizations:
                    row = pd.DataFrame([[pmid, authors[author_index], organizations[0], organizations[1],
                                        roles[author_index], affiliations[author_index]]],
                                       columns=['PMID', 'AuthorID', 'Author CHOP', 'Author PENN', 'ROLE', 'Affiliation'])
                    author_record_df = author_record_df.append(row, ignore_index=True)

            authors = ';'.join(authors)

            row = pd.DataFrame([[pmid, title, abstract, year, month, authors, mesh_term, date]],
                               columns=['PMID', 'Title', 'Abstract', 'Year', 'Month', 'Author List', 'Subject List',
                                        'Date'])
            paper_record_df = paper_record_df.append(row)

            title_list.append(title)
            abstract_list.append(abstract)

        except Exception as e:
            msg = 'Error while processing PMID={0}'.format(pmid)
            logging.getLogger('regular').debug(msg)
            msg = 'Exception message = {0}'.format(e)
            logging.getLogger('regular').debug(msg)

    # store the record in a file for processing
    dataset = dict()
    dataset['title'] = title_list
    dataset['abstracts'] = abstract_list
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(path_or_buf='record_results/titles_abstracts.csv', index=False)

    # read the records from the file
    # dataset = pd.read_csv('record_results/titles_abstracts.csv')

    topic_modeling(dataset=dataset)

    pandas.io.formats.excel.header_style = None
    # contains all the metadata elements on the author level: Pubmed unique Identifier number(PMID), AuthorID (as a
    # combination of the author’s last name, first name, and initials), institution: chop=0, Penn=1, Role: Chief Author
    # (CA) Ordinary Author (OA) or Principal Author (PA) and the author's affiliation
    author_record_df.to_excel('record_results/author_record.xlsx', sheet_name='author_record', index=False)
    # contains all the metadata elements on the paper level: Pubmed unique Identifier number(PMID), Title, Abstract,
    # Year, Month, AuthorList, SubjectList, date
    paper_record_df.to_excel('record_results/paper_record.xlsx', sheet_name='paper_record', index=False)
    # contains all the metadata of the medical information: Pubmed unique Identifier number(PMID), Primary Medical
    # Subject Header (MESH) and the description ID
    medical_record_df.to_excel('record_results/medical_record.xlsx', sheet_name='medical_record', index=False)

    logging.getLogger('line.regular.time.line').info('Recommendation System script finished running successfully.')


if __name__ == '__main__':
    main()



