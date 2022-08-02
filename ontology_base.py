import pandas as pd
from franz.openrdf.connect import ag_connect
from franz.openrdf.sail.allegrographserver import AllegroGraphServer
from franz.openrdf.repository.repository import Repository
from franz.openrdf.rio.rdfformat import RDFFormat
from franz.openrdf.query.query import QueryLanguage
import os

AGRAPH_HOST: str = 'localhost'
AGRAPH_PORT: int = 10035
AGRAPH_USER: str = 'test'
AGRAPH_PASSWORD: str = 'xyzzy'
prefix: str = r"<http://www.semanticweb.org/кристина/ontologies/2021/3/untitled-ontology-8#>"


class CreateServer(AllegroGraphServer):
    def __init__(self, catalog: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = self.openCatalog(catalog)
        self.mode = Repository.OPEN
        self.conn = None
        self.my_repository = None

    def access_repo(self, repo_name: str):

        self.my_repository = self.catalog.getRepository(repo_name, self.mode)
        self.my_repository.initialize()
        self.conn = self.my_repository.getConnection()

    def check_object(self, query_ask: str = "ASK WHERE {?s ?p ?o .}"):

        query = self.conn.prepareBooleanQuery(query=query_ask)

        return query.evaluate()

    def find_object(self, query_request: str = "SELECT ?s ?p ?o  WHERE {?s ?p ?o . }"):

        with self.conn.executeTupleQuery(query_request) as result:
            df = result.toPandas()

        return df

    def quit(self):

        if self.conn is not None:
            self.conn.close()

        if self.my_repository is not None:
            self.my_repository.shutDown()


if __name__ == '__main__':
    server = CreateServer(host=AGRAPH_HOST, port=AGRAPH_PORT,
                          user=AGRAPH_USER, password=AGRAPH_PASSWORD)
    server.access_repo('repo')

    df: pd.DataFrame = server.find_object(f"prefix : {prefix}"
                                          "SELECT *"
                                          "WHERE { "
                                          "?s ?p ?o ."
                                          " FILTER (?o IN (:int, :float ) )"
                                          "}")

    print(server.check_object())
    print(df['s'].values)

    server.quit()
