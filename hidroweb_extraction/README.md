# Hidroweb-API
Scripts próprios produzidos para a coleta de dados através da REST API do HidroWeb e armazenamento em um banco de dados.

Extrai dados dos principais endpoints disponíveis em https://www.ana.gov.br/hidrowebservice/swagger-ui/index.html#/

Os scripts dividem as datas de coletas (nos endpoints na qual a mesma é relevante) em batches de 30 dias, para contornar a limitação de 30 dias de consulta na API;

# Utilização do scrapper

## 1) Endpoints comuns (sem station code)

Observação: ao iniciar qualquer execução, o script garante automaticamente a criação do arquivo DuckDB (`data/hidroweb_scrapping.db`) quando ele ainda não existir.

Executa todos os endpoints comuns:

```
python3 endpoint_common.py --endpoints all
```

Executa apenas endpoints específicos:

```
python3 endpoint_common.py --endpoints HidroUF HidroBacia
```

Controle do modo de escrita no banco:

- `--inplace`: sobrescreve a tabela de destino (padrão);
- `--no-inplace`: faz append na tabela de destino;

Exemplo com append:

```
python3 endpoint_common.py --endpoints HidroUF --no-inplace
```

## 2) Endpoints chunkados (com station code e janela de datas)

Executa todos os endpoints chunkados para uma estação:

```
python3 endpoint_chunk.py --endpoints all --stationcode 87450004
```

Executa endpoint específico para múltiplas estações (lista por espaço):

```
ython3 endpoint_chunk.py --endpoints HidroinfoanaSerieTelemetricaDetalhada_v1 --stationcode 87450004 87444000 87399000
```

Também aceita lista separada por vírgula:

```
python3 endpoint_chunk.py --endpoints HidroinfoanaSerieTelemetricaAdotada_v2 --stationcode 87450004,87444000,87399000
```

Executa com janela de datas customizada:

```
python3 endpoint_chunk.py --endpoints all --stationcode 87450004 87444000 --start-date 2020-01-01 --end-date 2020-12-31
```

Controle do modo de escrita no banco:

- `--inplace`: sobrescreve a tabela de destino (padrão);
- `--no-inplace`: faz append na tabela de destino;

## 3) Consultas na base de dados;
Para visualizar a base de dados no terminal, usar o 'duckdb-cli'. Na pasta do projeto, ativar o venv e rodar duckdb no path da base
```
source ./venv/bin/activate
duckdb data/hidroweb_scrapping.db
```

Usar .help para obter a lista de comandos. Queries em SQL podem ser executadas diretamente no duckdb-cli, sempre seguidas de um semicolon ';', e.g.:
```
SELECT * FROM table_name;
```

Para executar consultas rápidas na base de dados localizada em ./data/hidroweb_scrapping.db, instanciar a classe de interação em 'db_handler.py' e usar o método '.run()':
```
python3 -c "from db_handler import DBConnection; db = DBConnection(); print(db.run(query='SELECT * FROM HidroBacia')['result'])"
```

Para salvar esses resultados em '.csv', adicionar o '.to_csv()' no comando anterior:
```
python3 -c "from db_handler import DBConnection; db = DBConnection(); db.run('SELECT * FROM table_name')['result'].to_csv('exemple.csv', index=False)"
```

## 4) Uso da base de dados com DBeaver;
Pela base ser DuckDB, é preciso baixar ela localmente com 'scp' e conectar o DBeaver nela;

# Mapeamento das Tabelas
| Nome Tabela | Endpoint | Conteúdo |
| --- | --- | --- |
| HidrosatSerieDados | /EstacoesTelemetricas/HidrosatSerieDados/v1 | Séries das estações virtuais (HidroSat). |
| HidrosatInventarioEstacoes | /EstacoesTelemetricas/HidrosatInventarioEstacoes/v1 | Inventário de estações virtuais (estimação por satélite) cadastradas na base HidroSat. |
| HidroinfoanaSerieTelemetricaDetalhada | /EstacoesTelemetricas/HidroinfoanaSerieTelemetricaDetalhada/v1 | Séries das estações telemétricas com dados adotados e dados brutos disponíveis. |
| HidroinfoanaSerieTelemetricaDetalhada | /EstacoesTelemetricas/HidroinfoanaSerieTelemetricaDetalhada/v2 | Séries das estações telemétricas com dados adotados e dados brutos disponíveis. |
| HidroinfoanaSerieTelemetricaAdotada | /EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1 | Séries das estações telemétricas com dados adotados de chuva, nível e vazão. |
| HidroinfoanaSerieTelemetricaAdotada | /EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v2 | Séries das estações telemétricas com dados adotados de chuva, nível e vazão. |
| HidroUF | /EstacoesTelemetricas/HidroUF/v1 | Lista de unidades federativas cadastradas na base HIDRO. |
| HidroSubBacia | /EstacoesTelemetricas/HidroSubBacia/v1 | Lista de sub-bacias hidrográficas cadastradas na base HIDRO. |
| HidroSerieVazao | /EstacoesTelemetricas/HidroSerieVazao/v1 | Séries de vazão das estações convencionais (coleta manual). |
| HidroSerieSedimentos | /EstacoesTelemetricas/HidroSerieSedimentos/v1 | Séries de sedimento das estações convencionais (coleta manual). |
| HidroSerieResumoDescarga | /EstacoesTelemetricas/HidroSerieResumoDescarga/v1 | Séries de medições de descarga líquida das estações. |
| HidroSerieQA | /EstacoesTelemetricas/HidroSerieQA/v1 | Séries de qualidade de água das estações convencionais (coleta manual). |
| HidroSeriePerfilTransversal | /EstacoesTelemetricas/HidroSeriePerfilTransversal/v1 | Séries de medições do perfil transversal das estações. |
| HidroSerieGranulometria | /EstacoesTelemetricas/HidroSerieGranulometria/v1 | Série de granulometria para as estações. |
| HidroSerieCurvaDescarga | /EstacoesTelemetricas/HidroSerieCurvaDescarga/v1 | Série de curvas de descarga líquida traçadas para as estações. |
| HidroSerieCotas | /EstacoesTelemetricas/HidroSerieCotas/v1 | Séries de cota das estações convencionais (coleta manual). |
| HidroSerieChuva | /EstacoesTelemetricas/HidroSerieChuva/v1 | Séries de chuva das estações convencionais (coleta manual). |
| HidroRio | /EstacoesTelemetricas/HidroRio/v1 | Lista de corpos hídricos cadastrados na base HIDRO. |
| HidroMunicipio | /EstacoesTelemetricas/HidroMunicipio/v1 | Lista de municípios cadastrados na base HIDRO (com código diferente do IBGE). |
| HidroInventarioEstacoes | /EstacoesTelemetricas/HidroInventarioEstacoes/v1 | Inventário completo de estações cadastradas na base Hidro. |
| HidroEntidade | /EstacoesTelemetricas/HidroEntidade/v1 | Lista de entidades cadastradas na base Hidro (responsável e operador das estações). |
| HidroBacia | /EstacoesTelemetricas/HidroBacia/v1 | Lista de bacias hidrográficas cadastradas na base Hidro. |

# Estações coletadas
-> Estações Fluviométricas, em operação, na Bacia 'ATLÂNTICO, TRECHO SUDESTE', no RS:

```
nohup python3 endpoint_chunk.py --endpoints HidroinfoanaSerieTelemetricaDetalhada_v1 --no-inplace --start-date '2016-01-01' --end-date '2026-04-22' --stationcode 84420000 84991000 84991100 84991500 84992000 85001500 85029000 85029500 85040000 85050001 85050070 85050100 85050200 85050300 85050350 85051000 85068000 85074000 85076000 85080000 85080001 85080010 85101000 85110000 85130000 85130400 85140000 85140010 85161020 85170000 85170100 85170150 85180000 85189400 85219300 85233700 85233800 85234250 85234300 85234750 85235000 85235250 85240000 85259650 85259950 85260001 85260110 85279600 85279900 85290000 85300000 85300650 85300700 85306000 85309000 85310000 85311000 85349000 85350000 85365000 85365500 85379990 85380500 85382000 85395100 85395300 85400000 85400010 85401000 85425000 85427000 85430000 85435000 85436200 85438000 85438500 85438510 85439000 85439100 85439200 85442000 85466000 85470000 85480000 85480010 85500010 85500100 85500990 85570000 85600000 85610000 85620500 85623000 85623010 85642000 85642005 85642010 85644000 85645000 85653000 85658000 85662000 85730800 85730900 85735000 85739810 85810000 85830000 85850500 85900000 85910000 85911000 85930000 85940000 86060010 86095000 86099000 86100000 86100600 86100800 86101000 86102000 86109000 86110000 86117000 86118000 86125000 86125050 86125100 86125120 86125130 86125160 86125165 86125200 86125500 86160000 86162000 86163000 86194900 86195000 86195100 86200500 86200900 86210900 86220600 86260000 86280500 86281000 86298000 86305000 86306000 86321000 86329000 86350000 86390900 86402000 86403000 86404100 86404200 86405000 86406000 86410000 86410050 86410800 86420000 86430500 86430900 86447000 86448000 86450000 86450500 86450600 86451000 86470800 86470900 86471000 86472000 86472500 86472600 86479000 86480000 86487000 86488000 86489000 86493000 86495500 86497300 86497400 86500000 86503800 86504900 86505400 86505500 86506000 86507000 86510000 86518300 86518500 86519500 86519550 86519700 86519730 86520000 86520050 86520090 86520100 86529500 86555800 86560000 86580000 86595000 86718000 86720000 86743000 86743700 86743800 86743900 86743950 86744110 86745000 86746000 86746680 86748001 86780000 86788000 86800001 86855000 86879000 86879300 86880030 86880050 86880600 86881000 86888000 86895000 86900000 86950000 86996000 87010000 87020000 87040100 87068000 87070000 87072000 87075000 87078000 87100000 87101050 87101400 87101500 87103000 87107000 87109801 87109900 87110000 87110050 87110100 87120000 87120100 87150000 87153000 87160000 87160100 87163000 87165001 87168000 87168510 87168590 87168600 87170000 87189000 87199980 87200000 87200100 87202000 87219950 87228450 87230000 87231100 87241000 87242000 87242030 87255500 87260000 87270000 87271100 87290000 87294000 87300000 87301100 87309000 87309010 87311000 87316000 87317010 87317015 87317021 87317030 87317035 87317036 87317060 87317140 87317150 87317160 87317180 87317600 87318000 87318500 87318700 87332500 87333000 87337000 87337010 87350000 87351000 87360450 87361000 87361010 87361090 87361100 87366500 87374000 87375500 87376000 87376800 87377400 87377800 87380000 87380015 87380030 87381800 87382000 87382010 87382020 87382025 87385000 87385040 87390060 87390070 87393000 87398500 87398700 87398750 87398800 87398900 87398950 87398980 87399000 87401750 87403000 87405500 87406000 87406900 87409000 87409900 87410200 87420100 87420300 87420350 87420450 87421000 87422000 87423000 87424000 87425000 87442000 87444000 87446000 87450004 87450005 87450020 87450063 87450100 87460007 87460020 87460120 87460140 87460150 87460155 87460175 87460200 87460210 87500020 87510010 87510015 87510020 87510030 87510047 87510049 87510070 87510100 87540000 87550050 87580001 87599000 87670000 87785000 87790000 87795000 87800001 87904000 87905000 87915000 87920500 87920700 87921000 87955000 87970000 87980000 87990000 87991000 87991100 87992000 87995000 88020000 88060200 88060210 88070000 88150900 88176000 88177000 88181000 88188000 88260000 88300000 88300500 88365000 88365010 88399000 88549000 88550010 88575000 88640950 88641000 88643000 88644000 88690000 88690050 88700010 88710000 88750000 88810100 88840000 88850000 88900000 &
```