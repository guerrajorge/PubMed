from Bio import Entrez, Medline
import pandas as pd
import csv
import logging
from utils.logger import logger_initialization
from utils.parse import parse
import argparse


def output_author_information(info, auth_file, paper_file, med_file):

    logging.getLogger('regular').debug('pmid = {0}'.format(info['PMID']))
    logging.getLogger('regular').debug('title = {0}'.format(info['TI']))
    logging.getLogger('regular').debug('abstract = {0}'.format(info['AB']))
    logging.getLogger('regular').debug('publication type = {0}'.format(info['PT']))

    # loop through each of the authors and assign their roles and affiliations
    for author_index, author in enumerate(info['AUS']):
        role = ''
        # Assign the authors' roles
        # if less than 2 authors then they are considered "Chief Authors"
        if len(info['AUS']) <= 1:
            role = 'CA'
        # If a person is after the first two authors and it'snt the last author its considered
        # "Ordinary Author"
        elif author_index > 1 and author_index != len(info['AUS']):
            role = 'OA'
        # else "Principal Investigator)
        elif author_index == len(info['AUS']):
            role = 'PI'

        # split to check if multiple affiliations
        affiliations = info['AD'].split(';')

        # Assign the author organization
        for affiliation in affiliations:
            if 'children' in affiliation.lower():
                author_information['CHOP'].apped(1)
                author_information['PENN'].apped(0)
            elif 'perelman' in affiliation.lower() or 'school of medicine' in affiliation.lower() or \
                            'pennsylvania' in affiliation.lower():
                author_information['PENN'].apped(1)
                author_information['CHOP'].apped(0)

        auth_file.writeline(info['PMID'], info['AuthorID'], info['CHOP'], )


    logging.getLogger('regular').debug('publication type = {0}'.format(info['ROLE']))


def obtain_descriptions():
    # contains all the metadata elements on the author level: Pubmed unique Identifier number(PMID), AuthorID (as a
    # combination of the author’s last name, first name, and initials), institution: chop=0, Penn=1, Role: Chief Author
    # (CA) Ordinary Author (OA) or Principal Author (PA) and the author's affiliation
    author_record_df = pd.DataFrame(columns=['PMID', 'AuthorID', 'CHOP', 'PENN', 'ROLE', 'Affiliation'])
    # contains all the metadata elements on the paper level: Pubmed unique Identifier number(PMID), Title, Abstract,
    # Year, Month, AuthorList, SubjectList, date
    paper_record_df = pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Year', 'Month', 'AuthorList', 'SubjectList',
                                            'Date'])
    # contains all the metadata of the medical information: Pubmed unique Identifier number(PMID), Primary Medical
    # Subject Header (MESH) and the description ID
    medical_record_df = pd.DataFrame(columns=['PMID', 'MESH', 'Description'])

    # get the description, related to the MESH, in the 2017MeshTree.csv File
    mesh_tree_file_object = open(r'C:\Users\GUERRAMARJ\PycharmProjects\Pubmed\template\2017MeshTree.csv')
    file_reader = csv.reader(mesh_tree_file_object, delimiter=',')
    mesh_description_dict = dict()

    logging.getLogger('regular').info('processing each record and obtaining relevant information')
    for line in file_reader:
        # split_line[0] = Number, split_line[1] = Description and split_line[2] = MESH
        mesh_description_dict[line[2]] = line[1]
    mesh_tree_file_object.close()

    return author_record_df, paper_record_df, medical_record_df, mesh_description_dict


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
    #     out_handle.write(fetch_records_handle.read(validate=True))
    # # the results are now in the results.xml file and the original handle has had all of its data extracted
    # # (so we close it)
    # fetch_records_handle.close()

    logging.getLogger('regular').info('reading result files')
    records_handle = open("results.xml")
    fetch_records = parse(records_handle)

    # initializing variables
    mesh_description_dict = obtain_descriptions()

    # PMID=PubMed Unique Identifier, TI=Title, AB=Abstract, AD=Affiliation, FAU=Full Author, MH=MeSH Terms,
    # PT=Publication Type
    # for more information, look at the abbreviations in the /template/abbreviations.txt file
    author_information = {'PMID': '', 'TI': '', 'AB': '', 'FAU': '', 'AU': '', 'MH': '', 'PT': '', 'AD': ''}

    author_list = list()
    affiliation_list = list()
    mesh_list = list()

    first_record = True

    # get the relevant information for each record
    for record_index, line in enumerate(fetch_records):
        logging.getLogger('regular').debug('line index = {0}'.format(record_index))

        # remove new line delimiter
        line = line.replace('\n', '')

        # skip if empty string
        if not line:
            continue

        # getting the key (PMID, TITLE, ABSTRACT, etc) and its value
        key, value = line.split('- ')
        # remove spaces
        key.replace(' ', '')

        # check if key is relevant to the information of interest
        if key not in author_information.keys():
            continue

        if key == 'PMID':
            # if it is not the first record, that means that it is a new record and therefore needs to reset all the
            # variables
            if not first_record:
                author_information['AU'] = author_list
                author_information['AD'] = affiliation_list
                author_information['MH'] = mesh_list

                logging.getLogger('regular').debug('authors\' information = {0}'.format(author_information))

                # function to print's the author's information to the relevant files
                # output_author_information(author_information)

                author_information = dict['PMID':'', 'TI':'', 'AB':'', 'FAU':'', 'AU', 'ROLE':'', 'MH':'', 'PT':'',
                                          'AD':'']

                author_list = list()
                affiliation_list = list()

        # there might be multiple authors per PMID and therefore we need to add them to a list
        if key == 'FAU':
            author_list.append(value)
        # each author might have one or more affiliations
        elif key == 'AD':
            affiliation_list.append(value)
        # there might be multiple mesh terms
        elif key == 'MH':
            # some of the mesh terms might have an * that needs to be removed
            mesh_list.append(value.replace('*', ''))

        # add the authors' information
        author_information[key] = value

        # changing first record flag
        first_record = False

    logging.getLogger('line.regular.time.line').info('Recommendation System script finished running successfully.')


if __name__ == '__main__':
    main()



