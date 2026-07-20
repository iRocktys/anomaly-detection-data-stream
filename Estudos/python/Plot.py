import numpy as np
import matplotlib.pyplot as plt

class DspotPlot:
    def plot(
        caudaValores,
        resultadoSerie,
        residuosCalibracao,
        mediasLocaisCalibracao,
        initialResidualThreshold,
        initialQuantile,
        driftDepth,
        colunaSelecionada="score",
        figsize=(20, 8),
    ):
        caudaValores = np.asarray(
            caudaValores,
            dtype=float,
        )

        residuosCalibracao = np.asarray(
            residuosCalibracao,
            dtype=float,
        )

        mediasLocaisCalibracao = np.asarray(
            mediasLocaisCalibracao,
            dtype=float,
        )

        quantidadeCauda = len(caudaValores)
        quantidadeSerie = len(resultadoSerie)
        quantidadeTotal = quantidadeCauda + quantidadeSerie

        indiceCompleto = np.arange(
            quantidadeTotal
        )

        valoresIncrementais = resultadoSerie[
            "valor"
        ].to_numpy(dtype=float)

        valoresCompletos = np.concatenate([
            caudaValores,
            valoresIncrementais,
        ])

        mediaLocalCompleta = np.full(
            quantidadeTotal,
            np.nan,
            dtype=float,
        )

        mediaLocalCompleta[
            driftDepth:quantidadeCauda
        ] = mediasLocaisCalibracao

        mediaLocalCompleta[
            quantidadeCauda:
        ] = resultadoSerie[
            "mediaLocal"
        ].to_numpy(dtype=float)

        limiarTCompleto = (
            mediaLocalCompleta
            + initialResidualThreshold
        )

        limiarDspotCompleto = np.full(
            quantidadeTotal,
            np.nan,
            dtype=float,
        )

        limiarDspotCompleto[
            quantidadeCauda:
        ] = resultadoSerie[
            "limiarExtremo"
        ].to_numpy(dtype=float)

        indicesPicosCalibracao = (
            driftDepth
            + np.flatnonzero(
                residuosCalibracao
                > initialResidualThreshold
            )
        )

        valoresPicosCalibracao = valoresCompletos[indicesPicosCalibracao]
        mascaraPicosIncrementais = (resultadoSerie["classificacao"] == "pico")
        mascaraAnomaliasIncrementais = (resultadoSerie["classificacao"] == "anomalia")
        indicesPicosIncrementais = resultadoSerie.loc[mascaraPicosIncrementais, "indiceGlobal"].to_numpy(dtype=int)
        valoresPicosIncrementais = resultadoSerie.loc[mascaraPicosIncrementais, "valor"].to_numpy(dtype=float)
        indicesAnomalias = resultadoSerie.loc[mascaraAnomaliasIncrementais, "indiceGlobal"].to_numpy(dtype=int)
        valoresAnomalias = resultadoSerie.loc[mascaraAnomaliasIncrementais, "valor"].to_numpy(dtype=float)

        fig, axis = plt.subplots(figsize=figsize)
        axis.axvspan(
            0,
            quantidadeCauda - 1,
            color="lightgray",
            alpha=0.35,
            label="Calibração da cauda",
            zorder=0,
        )

        axis.plot(
            indiceCompleto,
            valoresCompletos,
            color="tab:blue",
            label="Série",
            zorder=2,
        )

        axis.plot(
            indiceCompleto,
            mediaLocalCompleta,
            color="green",
            label="Média local",
            zorder=6,
        )

        axis.plot(
            indiceCompleto,
            limiarTCompleto,
            color="tab:orange",
            label=f"Limiar T ({initialQuantile:.2f})",
            zorder=4,
        )

        axis.plot(
            indiceCompleto,
            limiarDspotCompleto,
            color="red",
            label="Limiar DSPOT",
            zorder=5,
        )

        axis.scatter(
            indicesPicosCalibracao,
            valoresPicosCalibracao,
            color="grey",
            marker="o",
            s=75,
            label="Picos Calibração",
            zorder=6,
        )

        axis.scatter(
            indicesPicosIncrementais,
            valoresPicosIncrementais,
            color="black",
            marker="o",
            s=75,
            label="Picos Incrementais",
            zorder=6,
        )

        axis.scatter(
            indicesAnomalias,
            valoresAnomalias,
            color="red",
            marker="x",
            s=100,
            linewidth=2.8,
            label="Anomalias",
            zorder=7,
        )

        axis.set_title(
            "DSPOT",
            fontsize=17,
            fontweight="bold",
        )

        axis.set_xlabel("Instância")
        axis.set_ylabel(colunaSelecionada)

        axis.set_xlim(
            0,
            quantidadeTotal - 1,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=6,
            frameon=False,
            fontsize=11,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)
        plt.show()