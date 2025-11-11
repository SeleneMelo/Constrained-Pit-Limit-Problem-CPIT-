import pandas as pd
import numpy as np
import argparse
import random
import time
import os

# Classe MineLibCPIT: gerencia leitura de dados e estruturação

class MineLibCPIT:

    def __init__(self, arquivo_csv: str):
        self.arquivo_csv = arquivo_csv
        self.df = pd.read_csv(arquivo_csv)
        self._validar_colunas()
        self._preparar_dados()

    def _validar_colunas(self):
        colunas_necessarias = {"id", "x", "y", "z", "tonn", "dest", "val_ore", "precedentes"}
        faltando = colunas_necessarias - set(self.df.columns)
        if faltando:
            raise ValueError(f"Arquivo CSV faltando colunas: {faltando}")

    def _validar_colunas(self):
        colunas_basicas = {"id", "x", "y", "z", "precedentes"}
        faltando = colunas_basicas - set(self.df.columns)
        if faltando:
            raise ValueError(f"Arquivo CSV faltando colunas: {faltando}")

    def _preparar_dados(self):
        #fallback automatico
        self.ids = list(self.df["id"])
        self.tonn_dict = dict(zip(self.df["id"], self.df["tonn"]))
        self.dest_dict = dict(zip(self.df["id"], self.df["dest"]))
        self.val_ore_dict = dict(zip(self.df["id"], self.df["val_ore"]))
        self.preds_dict = {row.id: eval(row.precedentes) if isinstance(row.precedentes, str) else row.precedentes
                           for _, row in self.df.iterrows()}
        self.succs_dict = self._calcular_sucessores()
        self.df["tonn"] = self.df.get("tonn", 100)
        self.df["val_ore"] = self.df.get("val_ore", (self.df["z"].max() - self.df["z"]) * 10)
        self.df["dest"] = self.df.get("dest", 1)


    def _calcular_sucessores(self):
        succs = {b: [] for b in self.ids}
        for b, preds in self.preds_dict.items():
            for p in preds:
                succs[p].append(b)
        return succs


# Classe CPITSolverCompleto: heurísticas, GA e baseline

class CPITSolverCompleto:
    def __init__(self, instancia: MineLibCPIT, pop_size=50, num_geracoes=50, taxa_mutacao=0.05, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.inst = instancia
        self.pop_size = pop_size
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.melhor_solucao = None
        self.melhor_vpl = -np.inf
        self.historico = []

    # ----------------- Avaliação de soluções -----------------

    def calcular_vpl(self, solucao, taxa_desconto=0.15, capacidade=10_000_000):
        ano = 1
        ton_acumulada = 0.0
        valor_total = 0.0
        visitados = set()

        for b in solucao:
            preds = self.inst.preds_dict[b]
            if all(p in visitados for p in preds):
                dest = self.inst.dest_dict[b]
                ton = float(self.inst.tonn_dict[b])
                val = float(self.inst.val_ore_dict[b])

                # Cálculo de valor líquido compatível com baseline
                if dest == 1:  # minério
                    # -0.75 * tonn + val_ore (assumindo val_ore = process_profit)
                    cash = -0.75 * ton + val
                    if ton_acumulada + ton > capacidade:
                        ano += 1
                        ton_acumulada = 0
                    ton_acumulada += ton
                else:  # estéril
                    cash = -0.75 * ton  # sem process profit

                # Desconto compatível com baseline (ano-1)
                valor_total += cash / ((1 + taxa_desconto) ** (ano - 1))
                visitados.add(b)

        return valor_total


    # ----------------- Verificação e reparo ------------------

    def verificar_factibilidade(self, solucao):
        visitados = set()
        for b in solucao:
            if any(pred not in visitados for pred in self.inst.preds_dict[b]):
                return False
            visitados.add(b)
        return True

    def reparar_solucao(self, solucao):
        solucao_corrigida = []
        visitados = set()
        for b in solucao:
            preds = self.inst.preds_dict[b]
            for p in preds:
                if p not in visitados:
                    solucao_corrigida.append(p)
                    visitados.add(p)
            if b not in visitados:
                solucao_corrigida.append(b)
                visitados.add(b)
        return solucao_corrigida

    # ----------------- População inicial ---------------------

    def gerar_solucao_aleatoria(self):
        solucao = self.inst.ids.copy()
        random.shuffle(solucao)
        if not self.verificar_factibilidade(solucao):
            solucao = self.reparar_solucao(solucao)
        return solucao

    def gerar_populacao_inicial(self):
        return [self.gerar_solucao_aleatoria() for _ in range(self.pop_size)]

    # ----------------- Operadores genéticos ------------------

    def crossover(self, pai1, pai2):
        n = len(pai1)
        a, b = sorted(random.sample(range(n), 2))
        filho = pai1[a:b] + [x for x in pai2 if x not in pai1[a:b]]
        if not self.verificar_factibilidade(filho):
            filho = self.reparar_solucao(filho)
        return filho

    def mutacao(self, solucao):
        sol = solucao.copy()
        if random.random() < self.taxa_mutacao:
            i, j = random.sample(range(len(sol)), 2)
            sol[i], sol[j] = sol[j], sol[i]
            if not self.verificar_factibilidade(sol):
                sol = self.reparar_solucao(sol)
        return sol

    # ----------------- Algoritmo Genético --------------------

    def executar(self):
        pop = self.gerar_populacao_inicial()
        fitness = [self.calcular_vpl(sol) for sol in pop]
        self.melhor_solucao = pop[np.argmax(fitness)]
        self.melhor_vpl = max(fitness)

        for g in range(self.num_geracoes):
            nova_pop = []
            while len(nova_pop) < self.pop_size:
                pais = random.sample(pop, 2)
                filho = self.crossover(pais[0], pais[1])
                filho = self.mutacao(filho)
                nova_pop.append(filho)

            fitness = [self.calcular_vpl(sol) for sol in nova_pop]
            melhor_gen = max(fitness)
            if melhor_gen > self.melhor_vpl:
                self.melhor_vpl = melhor_gen
                self.melhor_solucao = nova_pop[np.argmax(fitness)]

            self.historico.append(self.melhor_vpl)
            print(f"Geração {g+1}/{self.num_geracoes} — Melhor VPL: {self.melhor_vpl:.2f}")

        return self.melhor_solucao, self.melhor_vpl

    # ----------------- Baseline TopoSort ---------------------

    def baseline_toposort(self):
        visitados = set()
        ordem = []
        preds_restantes = {b: set(self.inst.preds_dict[b]) for b in self.inst.ids}
        while len(ordem) < len(self.inst.ids):
            livres = [b for b, preds in preds_restantes.items() if not preds and b not in visitados]
            if not livres:
                break
            b = random.choice(livres)
            ordem.append(b)
            visitados.add(b)
            for s in self.inst.succs_dict[b]:
                preds_restantes[s].discard(b)
        return ordem, self.calcular_vpl(ordem)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Solver CPIT com GA e baseline TopoSort.")
    parser.add_argument("--instancia", type=str, required=True, help="Caminho para o CSV da instância.")
    parser.add_argument("--pop", type=int, default=50, help="Tamanho da população.")
    parser.add_argument("--geracoes", type=int, default=50, help="Número de gerações.")
    parser.add_argument("--mutacao", type=float, default=0.05, help="Taxa de mutação.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    args = parser.parse_args()

    inicio = time.time()
    print(f"\nCarregando instância: {args.instancia}")
    instancia = MineLibCPIT(args.instancia)

    print("Executando baseline (TopoSort)...")
    solver = CPITSolverCompleto(instancia)
    base_sol, base_vpl = solver.baseline_toposort()
    print(f"Baseline TopoSort VPL = {base_vpl:.2f}")

    print("\nExecutando Algoritmo Genético...")
    solver = CPITSolverCompleto(instancia, args.pop, args.geracoes, args.mutacao, args.seed)
    sol, vpl = solver.executar()

    tempo = time.time() - inicio
    print(f"\nExecução finalizada em {tempo:.2f}s")
    print(f"Melhor VPL GA = {vpl:.2f}")
    print(f"Melhoria sobre baseline: {(vpl - base_vpl) / base_vpl * 100:.2f}%")

    # salva histórico
    saida = os.path.splitext(os.path.basename(args.instancia))[0]
    pd.DataFrame({"geracao": range(1, len(solver.historico) + 1), "VPL": solver.historico}).to_csv(f"historico_{saida}.csv", index=False)
    print(f"Histórico salvo em historico_{saida}.csv")


if __name__ == "__main__":
    main()
