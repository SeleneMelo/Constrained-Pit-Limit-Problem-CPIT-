Para rodar o baseline toposort:

  python3 baseline_toposort.py


Para rodar o algoritmo genético:

  python3 merge_precedences.py "Modelo de Blocos.csv" "Modelo_com_Precedencias.csv" "modelo_final.csv" 
  python3 genetic_algorithmCPIT.py --instancia modelo_final.csv --pop 80 --geracoes 100 --mutacao 0.08 --seed 123
