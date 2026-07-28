# Códigos robustos por framework - CICIoT2023 cenário binário

Arquivos gerados separadamente para cada framework/modelo do projeto:

1. Scikit-learn Random Forest
2. FLAML AutoML
3. LazyPredict
4. TPOT AutoML
5. H2O AutoML
6. PyCaret
7. Auto-PyTorch
8. Auto-Sklearn
9. AutoGluon
10. AutoKeras

## Estrutura comum

Todos os notebooks seguem a mesma estrutura do pipeline Auto-PyTorch:

- Leitura do CICIoT2023 a partir de `/content/drive/MyDrive/Dataset/CSV`;
- Leitura em chunks de 200.000 linhas;
- Exclusão dos arquivos `Merged*.csv`;
- Criação de `label_original`, `label_binary`, `label_grouped` e `label_multiclass`;
- Remoção de colunas de vazamento, incluindo `source_file` e rótulos antigos;
- Conversão de features para `float32`;
- Divisão 80% treino e 20% teste, com `stratify=True` e `random_state=42`;
- Configuração de limite de memória `150000 MB`;
- Avaliação final no teste externo;
- Salvamento individual do modelo com o nome do framework.

## Observações

- `MAX_ROWS_PER_CLASS = None` usa o máximo disponível. Para teste rápido, altere para `50_000`, `100_000` ou `500_000`.
- Auto-PyTorch e Auto-Sklearn podem exigir ambiente Conda/Python compatível no Colab.
- LazyPredict, TPOT, PyCaret e AutoKeras podem ser muito lentos no dataset completo. Para testes iniciais, recomenda-se limitar `MAX_ROWS_PER_CLASS`.
