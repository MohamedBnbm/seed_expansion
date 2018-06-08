# seed_expansion

Les donnees Twitter sont enregistrees dans des bases de donnees Redis et ElasticSearch sur un serveur.

Le programme telecharge les donnees du serveur et enregistre, pour chaque domaine(ai, politique, f1 ...), les resultats dans les fichiers suivants:

	- 'graph.txt' : representation du graphe avec les voisins de chaque noeud | {noeud : [voisins, valences]}
	- 'intensity.txt' : score d'importance des noeuds obtenu par marche aleatoire | {noeud : importance}
	- 'clusters.txt' : resultats du clustering obtenus avec l'algo Louvain | {cluster : [identifiants utilisateur]}
	- 'lists_clusters.txt' : identifiants des listes dans chaque cluster | {cluster : [identifiants liste]}
	- 'cluster_names.txt' : noms des listes dans chaque cluster | {cluster : [noms liste]}
	- 'clusters_kws.txt' : mots cles des noms de liste dans chaque cluster | {cluster : [mots cles]}
	- 'user_names_info.txt' : nom, importance et cluster de chaque noeud | {noeud : nom, importance, cluster}
	- 'graphe.gexf' : graphe en format gexf
	- 'desired_vertices.txt' : liste de noeuds apartenant aux clusters choisis par l'utilisateur pour une analyse plus fine
	- 'reduced_clusters_kws.txt' : clusters Louvain du graphe reduit et leurs mots cles | {cluster : [mots cles]}
	- 'graphe_reduit.gexf' : gaphe reduit en format gexf
