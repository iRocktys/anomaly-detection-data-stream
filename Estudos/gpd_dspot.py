from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq


class GeneralizedPareto:
    """Funções matemáticas da Distribuição Generalizada de Pareto."""

    @staticmethod
    def cdf(valores, gamma, sigma):
        valores = np.asarray(valores, dtype=float)

        if sigma <= 0:
            raise ValueError("Sigma deve ser maior que zero.")

        if np.any(valores < 0):
            raise ValueError("Os excessos devem ser não negativos.")

        if np.isclose(gamma, 0.0):
            return 1 - np.exp(-valores / sigma)

        suporte = 1 + gamma * valores / sigma
        resultado = np.full(valores.shape, np.nan, dtype=float)
        validos = suporte > 0
        resultado[validos] = 1 - suporte[validos] ** (-1 / gamma)

        return resultado

    @staticmethod
    def sobrevivencia(valores, gamma, sigma):
        return 1 - GeneralizedPareto.cdf(
            valores,
            gamma,
            sigma
        )

    @staticmethod
    def densidade(valores, gamma, sigma):
        valores = np.asarray(valores, dtype=float)

        if sigma <= 0:
            raise ValueError("Sigma deve ser maior que zero.")

        if np.any(valores < 0):
            raise ValueError("Os excessos devem ser não negativos.")

        if np.isclose(gamma, 0.0):
            return np.exp(-valores / sigma) / sigma

        suporte = 1 + gamma * valores / sigma
        resultado = np.full(valores.shape, np.nan, dtype=float)
        validos = suporte > 0

        resultado[validos] = (
            (1 / sigma)
            * suporte[validos] ** (-1 / gamma - 1)
        )

        return resultado

    @staticmethod
    def logverossimilhanca(valores, gamma, sigma):
        valores = np.asarray(valores, dtype=float)
        quantidade = len(valores)

        if quantidade == 0:
            return -np.inf

        if sigma <= 0:
            return -np.inf

        if np.any(valores < 0):
            return -np.inf

        if np.isclose(gamma, 0.0):
            return (
                -quantidade * np.log(sigma)
                - np.sum(valores) / sigma
            )

        suporte = 1 + gamma * valores / sigma

        if np.any(suporte <= 0):
            return -np.inf

        return (
            -quantidade * np.log(sigma)
            - (1 + 1 / gamma)
            * np.sum(np.log(suporte))
        )


class TailSelection:
    """Seleção do limiar intermediário, picos e excessos."""

    @staticmethod
    def selecionar(valores, quantil):
        valores = np.asarray(valores, dtype=float)

        if len(valores) == 0:
            raise ValueError("A série está vazia.")

        if not 0 < quantil < 1:
            raise ValueError("O quantil deve estar entre zero e um.")

        limiar = float(np.quantile(valores, quantil))
        mascara = valores > limiar
        picos = valores[mascara]
        excessos = picos - limiar

        return {
            "limiar": limiar,
            "mascara": mascara,
            "picos": picos,
            "excessos": excessos
        }


class GridSearchGPD:
    """Estimação didática dos parâmetros da GPD por busca em grade."""

    @staticmethod
    def estimar(
        excessos,
        gammaMinimo=-0.40,
        gammaMaximo=0.80,
        quantidadeGamma=121,
        fatorSigmaMinimo=0.10,
        fatorSigmaMaximo=3.00,
        quantidadeSigma=150
    ):
        excessos = np.asarray(excessos, dtype=float)

        if len(excessos) == 0:
            raise ValueError("Não existem excessos para estimar a GPD.")

        media = float(np.mean(excessos))
        desvio = (
            float(np.std(excessos, ddof=1))
            if len(excessos) > 1
            else media
        )

        escalaReferencia = max(
            media,
            desvio,
            1e-8
        )

        gradeGamma = np.linspace(
            gammaMinimo,
            gammaMaximo,
            quantidadeGamma
        )

        gradeSigma = np.linspace(
            escalaReferencia * fatorSigmaMinimo,
            escalaReferencia * fatorSigmaMaximo,
            quantidadeSigma
        )

        resultados = []

        for gamma in gradeGamma:
            for sigma in gradeSigma:
                logAtual = GeneralizedPareto.logverossimilhanca(
                    excessos,
                    gamma,
                    sigma
                )

                if np.isfinite(logAtual):
                    resultados.append({
                        "gamma": gamma,
                        "sigma": sigma,
                        "logVerossimilhanca": logAtual
                    })

        if len(resultados) == 0:
            raise RuntimeError(
                "Nenhuma combinação válida foi encontrada na grade."
            )

        tabela = (
            pd.DataFrame(resultados)
            .sort_values(
                "logVerossimilhanca",
                ascending=False
            )
            .reset_index(drop=True)
        )

        melhor = tabela.iloc[0]

        return {
            "gamma": float(melhor["gamma"]),
            "sigma": float(melhor["sigma"]),
            "logVerossimilhanca": float(
                melhor["logVerossimilhanca"]
            ),
            "resultados": tabela
        }


class GrimshawGPD:
    """Estimação dos parâmetros da GPD pela técnica de Grimshaw."""

    @staticmethod
    def funcaoU(x, valores):
        valores = np.asarray(valores, dtype=float)
        termos = 1 + x * valores

        if np.any(termos <= 0):
            return np.nan

        return float(np.mean(1 / termos))

    @staticmethod
    def funcaoV(x, valores):
        valores = np.asarray(valores, dtype=float)
        termos = 1 + x * valores

        if np.any(termos <= 0):
            return np.nan

        return float(
            1 + np.mean(np.log(termos))
        )

    @staticmethod
    def funcaoW(x, valores):
        u = GrimshawGPD.funcaoU(
            x,
            valores
        )

        v = GrimshawGPD.funcaoV(
            x,
            valores
        )

        if not np.isfinite(u) or not np.isfinite(v):
            return np.nan

        return u * v - 1

    @staticmethod
    def criarIntervalos(valores, epsilon=1e-8):
        valores = np.asarray(valores, dtype=float)

        if len(valores) == 0:
            raise ValueError("Não existem excessos.")

        if np.any(valores <= 0):
            raise ValueError(
                "O Grimshaw exige excessos estritamente positivos."
            )

        maior = float(np.max(valores))
        menor = float(np.min(valores))
        media = float(np.mean(valores))

        intervalos = [
            (
                -1 / maior + epsilon,
                -epsilon
            )
        ]

        if media > menor:
            positivoInferior = (
                2
                * (media - menor)
                / (media * menor)
            )

            positivoSuperior = (
                2
                * (media - menor)
                / (menor ** 2)
            )

            if positivoSuperior > positivoInferior:
                intervalos.append(
                    (
                        positivoInferior,
                        positivoSuperior
                    )
                )

        return intervalos

    @staticmethod
    def buscarRaizes(
        valores,
        intervalos,
        quantidadePontos=10000
    ):
        valores = np.asarray(valores, dtype=float)
        raizes = []

        for limiteInferior, limiteSuperior in intervalos:
            pontos = np.linspace(
                limiteInferior,
                limiteSuperior,
                quantidadePontos
            )

            valoresW = np.array([
                GrimshawGPD.funcaoW(
                    x,
                    valores
                )
                for x in pontos
            ])

            for indice in range(len(pontos) - 1):
                xEsquerda = pontos[indice]
                xDireita = pontos[indice + 1]

                wEsquerda = valoresW[indice]
                wDireita = valoresW[indice + 1]

                if not (
                    np.isfinite(wEsquerda)
                    and np.isfinite(wDireita)
                ):
                    continue

                if np.isclose(
                    wEsquerda,
                    0,
                    atol=1e-10
                ):
                    raizes.append(xEsquerda)
                    continue

                if wEsquerda * wDireita < 0:
                    try:
                        raiz = brentq(
                            lambda x: GrimshawGPD.funcaoW(
                                x,
                                valores
                            ),
                            xEsquerda,
                            xDireita
                        )

                        raizes.append(raiz)

                    except ValueError:
                        continue

        return sorted(
            set(
                np.round(
                    raizes,
                    decimals=12
                )
            )
        )

    @staticmethod
    def estimar(
        excessos,
        quantidadePontos=10000
    ):
        excessos = np.asarray(excessos, dtype=float)

        if len(excessos) == 0:
            raise ValueError(
                "Não existem excessos para ajustar a GPD."
            )

        if np.any(excessos <= 0):
            raise ValueError(
                "Os excessos devem ser estritamente positivos."
            )

        intervalos = GrimshawGPD.criarIntervalos(
            excessos
        )

        raizes = GrimshawGPD.buscarRaizes(
            excessos,
            intervalos,
            quantidadePontos=quantidadePontos
        )

        candidatos = []

        gammaExponencial = 0.0
        sigmaExponencial = float(
            np.mean(excessos)
        )

        logExponencial = (
            GeneralizedPareto.logverossimilhanca(
                excessos,
                gammaExponencial,
                sigmaExponencial
            )
        )

        candidatos.append({
            "x": 0.0,
            "gamma": gammaExponencial,
            "sigma": sigmaExponencial,
            "logVerossimilhanca": logExponencial,
            "origem": "caso exponencial"
        })

        for raiz in raizes:
            if np.isclose(raiz, 0):
                continue

            vRaiz = GrimshawGPD.funcaoV(
                raiz,
                excessos
            )

            gamma = vRaiz - 1
            sigma = gamma / raiz

            if sigma <= 0:
                continue

            suporte = (
                1
                + gamma
                * excessos
                / sigma
            )

            if np.any(suporte <= 0):
                continue

            logAtual = (
                GeneralizedPareto.logverossimilhanca(
                    excessos,
                    gamma,
                    sigma
                )
            )

            if not np.isfinite(logAtual):
                continue

            candidatos.append({
                "x": raiz,
                "gamma": gamma,
                "sigma": sigma,
                "logVerossimilhanca": logAtual,
                "origem": "raiz de w(x)"
            })

        tabela = (
            pd.DataFrame(candidatos)
            .sort_values(
                "logVerossimilhanca",
                ascending=False
            )
            .reset_index(drop=True)
        )

        melhor = tabela.iloc[0]

        return {
            "gamma": float(melhor["gamma"]),
            "sigma": float(melhor["sigma"]),
            "logVerossimilhanca": float(
                melhor["logVerossimilhanca"]
            ),
            "x": float(melhor["x"]),
            "origem": melhor["origem"],
            "raizes": raizes,
            "intervalos": intervalos,
            "candidatos": tabela
        }


class POTThreshold:
    """Cálculo do limiar extremo e classificação POT."""

    @staticmethod
    def calcularZq(
        limiar,
        gamma,
        sigma,
        quantidadeObservacoes,
        quantidadePicos,
        risco
    ):
        if sigma <= 0:
            raise ValueError(
                "Sigma deve ser maior que zero."
            )

        if quantidadeObservacoes <= 0:
            raise ValueError(
                "A quantidade de observações deve ser positiva."
            )

        if quantidadePicos <= 0:
            raise ValueError(
                "A quantidade de picos deve ser positiva."
            )

        if not 0 < risco < 1:
            raise ValueError(
                "O risco deve estar entre zero e um."
            )

        if np.isclose(gamma, 0.0):
            return (
                limiar
                + sigma
                * np.log(
                    quantidadePicos
                    / (
                        risco
                        * quantidadeObservacoes
                    )
                )
            )

        razao = (
            risco
            * quantidadeObservacoes
            / quantidadePicos
        )

        return (
            limiar
            + sigma / gamma
            * (
                razao ** (-gamma)
                - 1
            )
        )

    @staticmethod
    def classificar(
        valores,
        limiarIntermediario,
        limiarExtremo
    ):
        valores = np.asarray(valores, dtype=float)

        classificacao = np.full(
            len(valores),
            "comum",
            dtype=object
        )

        classificacao[
            (valores > limiarIntermediario)
            & (valores <= limiarExtremo)
        ] = "pico"

        classificacao[
            valores > limiarExtremo
        ] = "anomalia"

        return classificacao


@dataclass
class DSPOTConfig:
    tamanhoAquecimento: int = 1024
    janelaMedia: int = 50
    quantilInicial: float = 0.98
    risco: float = 0.001
    quantidadePontosGrimshaw: int = 2000
    atualizarACadaPicos: int = 1


class DSPOT:
    """Execução completa do DSPOT sobre uma série univariada."""

    def __init__(self, config=None):
        self.config = (
            config
            if config is not None
            else DSPOTConfig()
        )

        self.limiarT = None
        self.limiarZq = None

        self.gamma = None
        self.sigma = None
        self.logVerossimilhanca = None

        self.excessos = []
        self.historicoNormal = []
        self.historicoAjustes = []

        self.quantidadeObservacoes = 0
        self.quantidadePicos = 0
        self.picosDesdeAtualizacao = 0

        self.resultados = None
        self.resultadoInicial = None

    def calcularMediaLocal(self):
        if len(self.historicoNormal) == 0:
            return 0.0

        valoresJanela = self.historicoNormal[
            -self.config.janelaMedia:
        ]

        return float(
            np.mean(valoresJanela)
        )

    def adicionarAoHistoricoNormal(self, valor):
        self.historicoNormal.append(
            float(valor)
        )

        if (
            len(self.historicoNormal)
            > self.config.janelaMedia
        ):
            self.historicoNormal.pop(0)

    def calcularResiduosAquecimento(
        self,
        valoresAquecimento
    ):
        valoresAquecimento = np.asarray(
            valoresAquecimento,
            dtype=float
        )

        medias = []
        residuos = []

        self.historicoNormal = []

        for indice, valor in enumerate(
            valoresAquecimento
        ):
            if indice == 0:
                mediaLocal = float(valor)
            else:
                mediaLocal = (
                    self.calcularMediaLocal()
                )

            residuo = (
                float(valor)
                - mediaLocal
            )

            medias.append(mediaLocal)
            residuos.append(residuo)

            self.adicionarAoHistoricoNormal(
                valor
            )

        return (
            np.asarray(medias, dtype=float),
            np.asarray(residuos, dtype=float)
        )

    def ajustarCauda(self, fase, indice):
        ajuste = GrimshawGPD.estimar(
            self.excessos,
            quantidadePontos=(
                self.config.quantidadePontosGrimshaw
            )
        )

        self.gamma = ajuste["gamma"]
        self.sigma = ajuste["sigma"]
        self.logVerossimilhanca = ajuste[
            "logVerossimilhanca"
        ]

        self.limiarZq = POTThreshold.calcularZq(
            limiar=self.limiarT,
            gamma=self.gamma,
            sigma=self.sigma,
            quantidadeObservacoes=(
                self.quantidadeObservacoes
            ),
            quantidadePicos=(
                self.quantidadePicos
            ),
            risco=self.config.risco
        )

        self.historicoAjustes.append({
            "indice": indice,
            "fase": fase,
            "quantidadeObservacoes": (
                self.quantidadeObservacoes
            ),
            "quantidadePicos": (
                self.quantidadePicos
            ),
            "gamma": self.gamma,
            "sigma": self.sigma,
            "logVerossimilhanca": (
                self.logVerossimilhanca
            ),
            "limiarTResidual": self.limiarT,
            "limiarZqResidual": self.limiarZq,
            "origem": ajuste["origem"],
            "quantidadeRaizes": len(
                ajuste["raizes"]
            )
        })

    def inicializar(self, valoresAquecimento):
        valoresAquecimento = np.asarray(
            valoresAquecimento,
            dtype=float
        )

        if (
            len(valoresAquecimento)
            != self.config.tamanhoAquecimento
        ):
            raise ValueError(
                "A quantidade de valores não corresponde "
                "ao tamanho do aquecimento."
            )

        medias, residuos = (
            self.calcularResiduosAquecimento(
                valoresAquecimento
            )
        )

        self.quantidadeObservacoes = len(
            residuos
        )

        selecao = TailSelection.selecionar(
            residuos,
            self.config.quantilInicial
        )

        self.limiarT = selecao["limiar"]
        self.excessos = list(
            selecao["excessos"]
        )

        self.quantidadePicos = len(
            self.excessos
        )

        if self.quantidadePicos < 3:
            raise ValueError(
                "Foram encontrados poucos picos no aquecimento. "
                "Reduza o quantil inicial ou aumente o aquecimento."
            )

        self.ajustarCauda(
            fase="aquecimento",
            indice=(
                self.config.tamanhoAquecimento
                - 1
            )
        )

        self.resultadoInicial = {
            "medias": medias,
            "residuos": residuos,
            "mascaraPicos": selecao["mascara"],
            "picos": selecao["picos"],
            "excessos": selecao["excessos"]
        }

        return self.resultadoInicial

    def processarValor(self, valor, indice):
        mediaLocal = self.calcularMediaLocal()

        residuo = (
            float(valor)
            - mediaLocal
        )

        limiarCaudaOriginal = (
            mediaLocal
            + self.limiarT
        )

        limiarExtremoOriginal = (
            mediaLocal
            + self.limiarZq
        )

        self.quantidadeObservacoes += 1
        atualizouCauda = False

        if residuo > self.limiarZq:
            classificacao = "anomalia"

        elif residuo > self.limiarT:
            classificacao = "pico incremental"

            excesso = (
                residuo
                - self.limiarT
            )

            self.excessos.append(excesso)
            self.quantidadePicos += 1
            self.picosDesdeAtualizacao += 1

            self.adicionarAoHistoricoNormal(
                valor
            )

            if (
                self.picosDesdeAtualizacao
                >= self.config.atualizarACadaPicos
            ):
                self.ajustarCauda(
                    fase="incremental",
                    indice=indice
                )

                self.picosDesdeAtualizacao = 0
                atualizouCauda = True

        else:
            classificacao = "normal"

            self.adicionarAoHistoricoNormal(
                valor
            )

        return {
            "mediaLocal": mediaLocal,
            "residuo": residuo,
            "limiarCauda": limiarCaudaOriginal,
            "limiarExtremo": limiarExtremoOriginal,
            "classificacao": classificacao,
            "gamma": self.gamma,
            "sigma": self.sigma,
            "zqResidual": self.limiarZq,
            "atualizouCauda": atualizouCauda
        }

    def executar(self, valores):
        valores = np.asarray(
            valores,
            dtype=float
        )

        if (
            len(valores)
            <= self.config.tamanhoAquecimento
        ):
            raise ValueError(
                "A série precisa conter valores posteriores "
                "ao aquecimento."
            )

        valoresAquecimento = valores[
            :self.config.tamanhoAquecimento
        ]

        resultadoInicial = self.inicializar(
            valoresAquecimento
        )

        registros = []

        for indice in range(
            self.config.tamanhoAquecimento
        ):
            mediaLocal = resultadoInicial[
                "medias"
            ][indice]

            if resultadoInicial[
                "mascaraPicos"
            ][indice]:
                classificacao = "pico inicial"
            else:
                classificacao = (
                    "aquecimento normal"
                )

            registros.append({
                "indice": indice,
                "valor": valores[indice],
                "mediaLocal": mediaLocal,
                "residuo": resultadoInicial[
                    "residuos"
                ][indice],
                "limiarCauda": (
                    mediaLocal
                    + self.limiarT
                ),
                "limiarExtremo": (
                    mediaLocal
                    + self.limiarZq
                ),
                "classificacao": classificacao,
                "fase": "aquecimento",
                "gamma": self.gamma,
                "sigma": self.sigma,
                "zqResidual": self.limiarZq,
                "atualizouCauda": False
            })

        for indice in range(
            self.config.tamanhoAquecimento,
            len(valores)
        ):
            resultado = self.processarValor(
                valores[indice],
                indice
            )

            registros.append({
                "indice": indice,
                "valor": valores[indice],
                "mediaLocal": resultado[
                    "mediaLocal"
                ],
                "residuo": resultado[
                    "residuo"
                ],
                "limiarCauda": resultado[
                    "limiarCauda"
                ],
                "limiarExtremo": resultado[
                    "limiarExtremo"
                ],
                "classificacao": resultado[
                    "classificacao"
                ],
                "fase": "incremental",
                "gamma": resultado["gamma"],
                "sigma": resultado["sigma"],
                "zqResidual": resultado[
                    "zqResidual"
                ],
                "atualizouCauda": resultado[
                    "atualizouCauda"
                ]
            })

        self.resultados = pd.DataFrame(
            registros
        )

        return self.resultados

    def historicoAjustesDataFrame(self):
        return pd.DataFrame(
            self.historicoAjustes
        )

    def resumoClassificacoes(self):
        if self.resultados is None:
            raise RuntimeError(
                "Execute o DSPOT antes de solicitar o resumo."
            )

        return (
            self.resultados[
                "classificacao"
            ]
            .value_counts()
            .rename_axis("classificacao")
            .reset_index(name="quantidade")
        )
