from Bio import Entrez, Medline
import pandas as pd
import csv


def main():
    # Entrez (http://www.ncbi.nlm.nih.gov/Entrez) is a data retrieval system that provides users access to NCBI’s
    # databases such as PubMed, GenBank, GEO, and many others
    # Use the mandatory email parameter so the NCBI can contact you if there is a proble
    Entrez.email = "guerramarj@email.chop.edu"     # Always tell NCBI who you are
    handle = Entrez.esearch(db="pubmed", retmax=50000, idtype="esearch", mindate="2014/01/01", maxdate="2017/05/01",
                            term="Perelman School of Medicine[Affiliation] OR Children's Hospital of "
                                 "Philadelphia[Affiliation] OR University of Pennsylvania School of "
                                 "Medicine[Affiliation] OR School of Medicine University of Pennsylvania[Affiliation]",
                            usehistory="y")
    search_results = Entrez.read(handle)
    id_list = search_results["IdList"]

    # get all the record based on the PMIDs
    fetch_records_handle = Entrez.efetch(db="pubmed", id=id_list, retmode="text", rettype="medline")
    # Use the Bio.Medline module to parse records
    fetch_records = Medline.parse(fetch_records_handle)

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
    mesh_tree_file_object = open(r'C:\Users\GUERRAMARJ\Desktop\Projects\scholar_recommendation\2017MeshTree.csv')
    file_reader = csv.reader(mesh_tree_file_object, delimiter=',')
    mesh_description_dict = dict()

    for line in file_reader:
        # split_line[0] = Number, split_line[1] = Description and split_line[2] = MESH
        mesh_description_dict[line[2]] = line[1]
    mesh_tree_file_object.close()

    # get the relevant information for each record
    for record_index, record in enumerate(fetch_records):
        pmid = ''
        title = ''
        abstract = ''
        affiliation = ''
        author_id = ''
        role = ''
        mesh_term = ''

        try:
            pmid = record.get("PMID")
            title = record.get("TI")
            abstract = record.get("AB")

            # Note: Currently the record.get("AD") method returns a string regardless of the number of authors i.e. if
            # there are two author, it will return as a string both affiliations. As of result, this script has to
            # manually get the author information and their respective affiliations
            fetch_records_handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            record = Entrez.read(fetch_records_handle)
            authors_list = record['PubmedArticle'][0]['MedlineCitation']['Article']['AuthorList']

            for author_index, author in enumerate(authors_list):
                affiliation = author['AffiliationInfo'][0]['Affiliation']
                author_id = author['LastName'] + ', ' + author['ForeName'] + ', ' + author['Initials']

                # Assign the author organization
                # 1 = chop, 0 = penn
                if 'children' in affiliation.lower():
                    chop_penn = 1
                elif 'perelman' in affiliation.lower() or 'school of medicine' in affiliation.lower() or 'pennsylvania' in \
                        affiliation.lower():
                    chop_penn = 0
                else:
                    ValueError('Affiliation is neither CHOP or Penn')

                # Assign the author's rle
                # if less than 2 authors then they are considered "Chief Authors"
                if author_index <= 1:
                    role = 'CA'
                # If a person is after the first two authors and it'snt the last author its considered "Ordinary Author"
                elif author_index > 1 and author_index != len(authors_list):
                    role = 'OA'
                # else "Principal Investigator)
                elif author_index == len(authors_list):
                    role = 'PI'
                else:
                    ValueError('Wrong author role specified')

                # insert the author information into the dataframe for later processing
                author_record_df.loc[record_index] = [pmid, author_id, chop_penn, role, affiliation]

            # Medical Subject Headings (MESH)
            mesh_term = record.get("MH")
            if mesh_term is not None:
                # fetch the description from the description obtain from the 2017MeshTree file
                description = mesh_description_dict[mesh_term]
                # insert the values in the dataframe
                medical_record_df.append([pmid], mesh_term, description)

        except ValueError as error_message:
            msg = 'Problem while processing the following '

if __name__ == '__main__':
    main()



