from Bio import Entrez, Medline
import pandas as pd
import csv
import logging
from utils.logger import logger_initialization
import argparse


########################################################################################################################
# THIS CODE USED HTTP protocol to parse the xml file and therefore its performance varies depending on the internet    #
# connection. IT IS NOT RELIABLE!                                                                                      #
########################################################################################################################

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

    records_handle = open("results.xml")

    logging.getLogger('regular').info('creating parser record handle')
    # Use the Bio.Medline module to parse records
    fetch_records = Medline.parse(records_handle)

    # contains all the metadata elements on the author level: Pubmed unique Identifier number(PMID), AuthorID (as a
    # combination of the author’s last name, first name, and initials), institution: chop=0, Penn=1, Role: Chief Author
    # (CA) Ordinary Author (OA) or Principal Author (PA) and the author's affiliation
    author_record_df = pd.DataFrame(columns=['PMID', 'AuthorID', 'CHOP_PENN', 'ROLE', 'Affiliation'])
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

    # get the relevant information for each record
    for record_index, record in enumerate(fetch_records):
        logging.getLogger('regular').debug('record index = {0}'.format(record_index))
        # initialize all the variables
        pmid = ''
        title = ''
        abstract = ''
        affiliation = ''
        author_id = ''
        role = ''
        mesh_term = ''

        try:
            pmid = record.get('PMID')
            title = record.get('TI')
            abstract = record.get('AB')

            logging.getLogger('regular').debug('pmid = {0}'.format(pmid))
            logging.getLogger('regular').debug('title = {0}'.format(title))
            logging.getLogger('regular').debug('abstract = {0}'.format(abstract))
            # only used for debugging
            publication_type = record.get('PT')
            logging.getLogger('regular').debug('publication type = {0}'.format(publication_type))

            # Note: Currently the record.get("AD") method returns a string regardless of the number of authors i.e. if
            # there are two author, it will return as a string both affiliations. As of result, this script has to
            # manually get the author information and their respective affiliations
            fetch_records_handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            manual_record = Entrez.read(fetch_records_handle)
            try:
                if 'Book Chapter' in publication_type:
                    authors_list = manual_record['PubmedBookArticle'][0]['BookDocument']['AuthorList']
                else:
                    # author_list for Pudmed Article
                    authors_list = manual_record['PubmedArticle'][0]['MedlineCitation']['Article']['AuthorList']
            except:
                logging.getLogger('regular').debug('error while obtaining the authors\' list')
                continue

            for author_index, author in enumerate(authors_list):
                try:
                    affiliation = author['AffiliationInfo'][0]['Affiliation']
                    author_id = author['LastName'] + ', ' + author['ForeName'] + ', ' + author['Initials']

                    logging.getLogger('regular').debug('affiliation = {0}'.format(affiliation))
                    logging.getLogger('regular').debug('author id = {0}'.format(author_id))

                    # Assign the author organization
                    # 1 = chop, 0 = penn
                    chop_penn = None
                    if 'children' in affiliation.lower():
                        chop_penn = 1
                    elif 'perelman' in affiliation.lower() or 'school of medicine' in affiliation.lower() or  \
                            'pennsylvania' in affiliation.lower():
                        chop_penn = 0

                    logging.getLogger('regular').debug('chop_penn = {0}'.format(chop_penn))

                    # Assign the author's rle
                    # if less than 2 authors then they are considered "Chief Authors"
                    if author_index <= 1:
                        role = 'CA'
                    # If a person is after the first two authors and it'snt the last author its considered
                    # "Ordinary Author"
                    elif author_index > 1 and author_index != len(authors_list):
                        role = 'OA'
                    # else "Principal Investigator)
                    elif author_index == len(authors_list):
                        role = 'PI'
                    else:
                        ValueError('Wrong author role specified')

                    logging.getLogger('regular').debug('role = {0}'.format(role))

                    if chop_penn is not None:
                        # insert the author information into the dataframe for later processing
                        author_record_df.loc[record_index] = [pmid, author_id, chop_penn, role, affiliation]
                except (IndexError, KeyError):
                    # sometimes there wil be organizations on the authors list, in those cases, skip it
                    continue

            # Medical Subject Headings (MESH)
            # this can be a list
            mesh_term = record.get("MH")
            logging.getLogger('regular').debug('mesh term = {0}'.format(mesh_term))
            if mesh_term is not None:
                # fetch the description from the description obtain from the 2017MeshTree file
                if len(mesh_term) > 1:

                    # because there are mesh_term that are not part of the 2017MeshTree, we have to loop through all
                    # of the mesh_term until one works i.e. the first one found in the 2017MeshTree
                    for mesh in mesh_term:

                        try:
                            term = mesh

                            print('term = {0}'.format(term))
                            # cleaning string
                            if '/' in term:
                                term = term.split('/')[0]
                            if '*' in term:
                                term = term.replace('*', '')

                            logging.getLogger('regular').debug('term = {0}'.format(term))

                            description = mesh_description_dict[term]

                        except KeyError:
                            logging.getLogger('regular').debug('not found term = {0}'.format(term))
                            continue

                # insert the values in the dataframe
                medical_record_df.append([pmid, mesh_term, description])

            # insert the paper information in the paper record dataframe
            # paper_record_df.append([pmid, title, abstract, year, month, authors_list, subject_list, date)

        except ValueError as error_message:
            msg = 'Problem while processing the following '
            print(msg)
            print('error message = {0}'.format(error_message))

    logging.getLogger('line.regular.time.line').info('Recommendation System script finished running successfully.')


if __name__ == '__main__':
    main()



