# Classificação de distúrbios pulmonares em radiografias de tórax usando Redes Convolucionais

A inteligência artificial têm sido uma ativa área de pesquisa por várias décadas, entretanto ainda apresenta vários desafios. Algoritmos de aprendizagem de máquina rápidos e precisos são fundamentais numa ampla gama de aplicações tais como: carros autônomos, reconhecimento prático da fala, pesquisa eficaz na web e na classificação de imagens médicas. Dentro desse contexto, as redes neurais profundas (Deep Learning), mais especificamente as Redes Neurais Convolucionais, mostraram desempenho inovador em diversas tarefas sofisticadas, especialmente aquelas relacionadas às imagens. Como o campo da radiologia depende principalmente da extração de informações úteis a partir de imagens, é uma área de aplicação muito natural para o aprendizado profundo. Sendo assim, este projeto se propõem a investigar a utilização de Redes Neurais Convolucionais para auxiliar os profissionais radiologistas e médicos no diagnóstico de distúrbios pulmonares.

Essa pesquisa demonstra que a utilização das Redes Neurais Convolucionais  é  um  método  adequado  para  reconhecer  aexistência de distúrbios pulmonares em radiografias torácicas, sendo  uma  ferramenta  útil  para  otimizar  a  prática  clínica  de profissionais na área da radiologia, potencialmente otimizando o  tempo  e  o  custo  de  trabalho  do  profissional  e  garantindo ao paciente uma triagem mais adequada. Apesar do tamanho pequeno  dos  conjuntos  de  dados  utilizados,  técnicas  como transferência  de  aprendizado,  aumento  de  dados  e Ensemble Learning,  produziram  resultados  satisfatórios  para  classificação de doenças pulmonares de forma automatizada. Os modelos concebidos podem ser expandidos para a efetiva aplicação na área médica, em que a realização de treinamento nessas redes utilizando conjunto de dados mais robustos e consistentes tornaria-os capazes de lidar com a grande heterogeneidade existente nas imagens médicas, permitindo a produção de novas tecnologias para otimizar, de forma complementar, o trabalho dos profissionais da radiologia. Além disso, pesquisas futuras podem analisar a interpretabilidade e reusabilidade de algoritmos  inteligentes  na  área  da  pneumologia,  assim  como técnicas  de  extração  das  informações  úteis  nos  conjuntos  de imagens  públicas  de  forma  otimizada.  Dessa  forma,  é  possível  observar  que  as  redes  neurais  convolucionais  colaboram para o reconhecimento automático de distúrbios pulmonares e corroboram  para  a  expansão  da  pesquisa  no  campo  da  visão computacional utilizando imagens médicas.

O  presente  trabalho  foi  realizado  com  apoio  do  Conselho  Nacional  de  Desenvolvimento  Científico  e  Tecnológico (CNPq), através do programa PIBIC/CNPq-UFCG, e também pela  Coordenação  de  Aperfeiçoamento  de  Pessoal  de  Nível Superior (CAPES), por meio do processo 88881.507204/2020-01, aprovado no Edital CAPES no âmbito do Edital Emergencial Nº 12/2020. Agradeço a professora Luciana Veloso e ao mestrando Leo Araújo por suas imensas contribuições nesse trabalho.

### Resultados Coletados - Experimento 5

|Modelos|Acurácia|Precisão|Sensibilidade|Inferência(ms)|
|:-----:|:------:|:------:|:-----------:|:------------:|
|DenseNet121|72,01%|72,51%|72,01%|24,77|
|InceptionV3|72,97%|73,48%|72,97%|21,08|
|Xception|72,07%|72,48%|72,07%|31,64|
|InceptionResnetV2|71,34%|72,93%|71,34%|43,50|
|ResNet101V2|72,80%|73,67%|72,80%|45,00|
|VGG16|70,47%|72,05%|70,47%|70,47|
|MobileNetV2|70,26%|71,14%|70,26%|8,99|
|Média Ponderada|77,20%|77,47%|77,20%|239,46|
|Voto da Maioria|79,54%|80,21%|79,54%|241,74|

### Artigo do trabalho
