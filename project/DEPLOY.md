# Publicar no Streamlit Community Cloud

Guia do zero até o link público. O repositório já está preparado — os passos
abaixo são o que falta fazer fora dele.

## Antes de começar

Confira que a raiz do repositório (`ml_evaluator/`) tem estes arquivos. Eles já
foram criados; a lista serve para o caso de algo se perder em um `git add`
incompleto.

```
ml_evaluator/
  .gitignore                   ignora .venv, __pycache__ e secrets.toml
  .python-version              fixa o Python em 3.11
  .streamlit/config.toml       config lida pelo Streamlit Cloud
  requirements.txt             dependências (cópia da de project/)
  project/
    app.py                     arquivo principal do app
    .streamlit/config.toml     mesma config, para rodar de dentro de project/
    requirements.txt
    ...
```

**Por que dois `config.toml` e dois `requirements.txt`?** O Streamlit lê
`.streamlit/config.toml` a partir do *diretório de trabalho*, não do diretório
do script. Rodando localmente de dentro de `project/`, ele lê o de `project/`;
no Cloud, que executa a partir da raiz do repositório, ele lê o da raiz. As
duas cópias precisam ser idênticas — `tests/test_deploy_config.py` falha se
divergirem.

## Passo 1 — Publicar o código no GitHub

O Streamlit Community Cloud só publica a partir do GitHub.

```bash
cd "C:\Users\Lucas Cardoso\Documents\Projects\ml_evaluator"

git init
git add .
git commit -m "ML Evaluator: ferramenta didatica de aprendizado supervisionado"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/ml-evaluator.git
git push -u origin main
```

Antes do `git push`, crie o repositório vazio em <https://github.com/new> —
sem README, sem .gitignore, para não gerar conflito no primeiro push.

Confira que `.venv/` **não** foi versionada:

```bash
git ls-files | Select-String ".venv"   # PowerShell; deve não retornar nada
```

O repositório pode ser público ou privado. No plano gratuito, apps a partir de
repositório privado contam contra uma cota menor.

## Passo 2 — Criar o app no Streamlit Cloud

1. Acesse <https://share.streamlit.io> e entre com a conta do GitHub.
2. Autorize o Streamlit a ler seus repositórios.
3. Clique em **Create app** e escolha **Deploy a public app from GitHub**.
4. Preencha:

   | Campo | Valor |
   | --- | --- |
   | Repository | `SEU_USUARIO/ml-evaluator` |
   | Branch | `main` |
   | Main file path | `project/app.py` |
   | App URL | o subdomínio que quiser, ex. `ml-evaluator` |

5. Em **Advanced settings**, selecione **Python 3.11**.
6. Clique em **Deploy**.

O primeiro build leva alguns minutos: o Cloud instala o `requirements.txt` e
sobe o app. O log aparece na própria tela — se algo falhar, é ali que o erro
aparece.

## Passo 3 — Conferir o resultado

Abra o link e verifique:

- A tela inicial é **"Como funciona"**.
- A aba do navegador mostra o ícone da árvore de decisão e o título
  "ML Evaluator — Aprendizado de Máquina".
- O tema está escuro.
- **Não** existe o menu de três pontos com opções de desenvolvedor
  (*Rerun*, *Clear cache*, *Record a screencast*).
- Faça um teste de ponta a ponta com `project/dados_exemplo/credito_classificacao.csv`.

## O parâmetro que remove o menu de desenvolvedor

Está em `.streamlit/config.toml`:

```toml
[client]
toolbarMode = "viewer"
```

| Valor | Efeito |
| --- | --- |
| `"auto"` | Padrão: mostra as opções para o desenvolvedor, esconde para o visitante |
| `"viewer"` | **Sempre escondido** — é o usado aqui |
| `"minimal"` | Só o que vier de fora ou de `st.set_page_config`; sem nada, some |
| `"developer"` | Sempre visível |

Junto vai `showErrorDetails = "none"`, que troca o traceback na tela por uma
mensagem genérica — o erro completo continua no log do servidor. Um traceback
assusta um público leigo sem informar nada útil a ele. Se precisar depurar em
produção, mude para `"full"`, faça o commit, espere o redeploy e devolva para
`"none"` depois.

Nota: o botão **"Manage app"**, no canto inferior direito, é do Streamlit Cloud
e só aparece para você, o dono do app. Visitantes não o veem, e ele não é
controlado por `toolbarMode`.

## Atualizações

Todo `git push` na branch publicada dispara redeploy automático. Se as
dependências mudarem, o Cloud reinstala o ambiente — isso demora mais.

Para forçar um redeploy limpo: **Manage app → Reboot app**.

## Se algo der errado

| Sintoma | Causa provável | Solução |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'scr'` | Main file path errado | Precisa ser `project/app.py` |
| `ModuleNotFoundError` de uma biblioteca | `requirements.txt` não está na raiz | Confirme que existe `ml_evaluator/requirements.txt` |
| Tema claro e menu de dev visível | Config não encontrada | Confirme que `ml_evaluator/.streamlit/config.toml` foi versionado (é um diretório oculto; `git add .streamlit` explicitamente) |
| Ícone da aba ausente | `assets/favicon.png` não versionado | `git add project/assets` |
| App "dorme" após dias sem uso | Comportamento normal do plano gratuito | Qualquer visita reativa em ~30 s |
| Erro de memória com arquivo grande | Limite de ~1 GB de RAM do plano gratuito | O app já limita upload a 25 MB; reduza `MAX_ROWS` em `scr/core/config.py` se necessário |

## Segredos

A aplicação **não usa nenhum segredo** — não há chave de API nem banco de dados,
e os dados enviados ficam apenas na memória da sessão.

Se um dia precisar de um: cadastre em **Manage app → Settings → Secrets** (formato
TOML) e leia com `st.secrets["NOME"]`. Nunca escreva a chave no código nem
versione `secrets.toml` — ele já está no `.gitignore`.

## Sobre privacidade, ao divulgar para alunos

Vale dizer isso a quem for usar: os arquivos enviados são processados na memória
do servidor do Streamlit Cloud e não são gravados em disco nem compartilhados
entre sessões, mas ainda assim trafegam por uma infraestrutura de terceiros.
Para uma aula, use dados sintéticos ou públicos — os exemplos em
`project/dados_exemplo/` foram gerados justamente para isso.
