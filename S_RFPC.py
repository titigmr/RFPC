import pandas as pd
import numpy as np
from fanalysis.pca import PCA
from statsmodels.api import OLS


def verification_bloc(blocs, vl, data):
    """
    Affiche les variables manifestes et les variables latentes associées.

    Paramètres :
    -----------

    blocs : la liste des indices des variables manifestes pour chacun des blocs
    vl : liste des noms des variables manifestes
    data : jeu de données

    """
    for enum, bloc in enumerate(blocs):
        bloc_var = data.iloc[:, bloc[0]:bloc[1]]
        vl_ = vl[enum]
        print("Bloc :", vl_, "\nVariables manifestes :", list(bloc_var.columns), "\n")


class RFPC:
    def __init__(self, data, blocs, vl, path):
        """
        Calcul l'approche RFPC pour les modèles à équation structurelles.

        Paramètres :
        ------------
        data : le jeu de donnée sous forme de DataFrame.
        blocs : la liste des indices des variables manifestes pour chaque bloc.
        vl : le nom des variables manifeste dans le même ordre que les blocs.
        path : matrice des liens, dans le même ordre que vl.
        """

        vl_data = self.__variable_latente__(data, blocs, vl)
        all_reg = self.__reg__(path)

        r2_all, self.list_resul_inner = self.__calcul_r2_inner__(all_reg, vl_data)
        self.final_inner, self.final_outer = self.__RFPC_estimate__(blocs, data, r2_all, vl)

        R2_mean = self.final_inner.R2[self.final_inner.R2 != 0].mean()
        AVE_mean = self.final_inner.AVE.mean()
        self.GOF = np.sqrt(R2_mean * AVE_mean)

    def __inner_outer__(self, data, begin, end, comp=False, name='Latent', r2=0, printing=False):
        """
        Estimation du modèle interne dans le cas de l'approche RFPC.


        Paramètres
        ----------
        data : dataframe contenant les données.
        begin, end : numéro de l'indice de début et de fin du bloc à estimer.
                    Un print affiche les variable pour vérifier.
        comp : par défaut False, si True, renvoit l'estimation de la variable latente.
        name : nom du bloc.
        r2 : par défaut 0. Si le bloc est endogène, renseigner le r2
            de la variable latente après estimation du modèle interne.
        """

        block = data.iloc[:, begin:end]
        f_pca = PCA(col_labels=list(data.iloc[:, begin:end].columns),
                    n_components=1)
        rfpc = f_pca.fit(np.array(data.iloc[:, begin:end]))
        if printing:
            print(rfpc.col_labels)

        comp_rfpc = pd.DataFrame(f_pca.row_coord_, columns=[name])
        if comp:
            return comp_rfpc

        loading = pd.concat([comp_rfpc, data.iloc[:, begin:end]],
                            axis=1).corr().iloc[1:, 0]
        weight = loading / f_pca.eig_[0]
        comm = loading ** 2
        redond = comm * r2

        out = np.array(pd.concat([weight, loading, comm, redond], axis=1))
        out_outer = pd.DataFrame(out, columns=["weight", "loading", "communality", "redundancy"],
                                 index=f_pca.col_labels)
        out_outer.insert(0, 'Bloc', name)

        AVE = out_outer.communality.mean()
        Mean_redundancy = AVE * r2

        out_inner = {"R2": r2, "AVE": AVE, "Mean_redundancy": Mean_redundancy}
        out_inner = np.round(pd.Series(out_inner), 3)

        return out_outer, out_inner

    def __variable_latente__(self, data, blocs, vl):
        """
        Calcul des variables latentes (1ère composante).

        """
        list_composante = []

        for enum, bloc in enumerate(blocs):
            composante = self.__inner_outer__(data, bloc[0], bloc[1], comp=True, name=str(vl[enum]))
            list_composante.append(composante)

        variable_latente = pd.concat(list_composante, axis=1)

        return variable_latente

    def __reg__(self, path):
        """
        Liste sous forme de dictionnaire les régressions du modèle interne.

        """
        all_reg = []

        for index in path.index:
            link = [path.loc[index, :] == 1][0]
            x = []
            names_index = list(path.index)
            for elem, name in zip(link, names_index):
                if elem:
                    x.append(name)
            y = index

            reg = {'y': y, 'x': x}
            all_reg.append(reg)

        return all_reg

    def __calcul_r2_inner__(self, all_reg, vl_data):
        """
        Calcule les R2 du modèle interne.
        """
        r2_all = []
        regression = []

        for i in all_reg:
            if len(i["x"]) == 0:
                r2 = 0
            else:
                r2 = OLS(endog=vl_data[i["y"]], exog=vl_data[i["x"]]).fit().rsquared
                coef = OLS(endog=vl_data[i["y"]], exog=vl_data[i["x"]]).fit().params
                se = OLS(endog=vl_data[i["y"]], exog=vl_data[i["x"]]).fit().bse
                tval = OLS(endog=vl_data[i["y"]], exog=vl_data[i["x"]]).fit().tvalues
                pval = OLS(endog=vl_data[i["y"]], exog=vl_data[i["x"]]).fit().pvalues

                regression.append({"coef": coef, "se": se, "tval": tval, "pval": pval})

            r2_all.append(r2)

        list_resul_inner = []

        for i in regression:
            result_inner = pd.DataFrame(i)
            list_resul_inner.append(result_inner)

        return r2_all, list_resul_inner

    def __RFPC_estimate__(self, blocs, data, r2_all, vl):
        """
        Estimation des modèles de mesure et de structure.
        Les retours donnent les indicateurs de performances et les coefficients.
        """

        outer_total = []
        inner_total = []

        for enum, bloc in enumerate(blocs):
            out_outer, out_inner = self.__inner_outer__(data, bloc[0], bloc[1],
                                                        comp=False, name=str(vl[enum]), r2=r2_all[enum], printing=True)

            outer_total.append(out_outer)
            inner_total.append(out_inner)

        final_outer = pd.concat(outer_total, axis=0)
        final_inner = pd.concat(inner_total, axis=1).T
        final_inner.index = list(vl)

        return final_inner, final_outer
