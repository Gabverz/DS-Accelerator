# Perguntas para o cliente (WeCogno / banco)

Lista acumulada de inconsistências encontradas na EDA. Enviar em um único e-mail.

## base_cadastral

1. `renda_anual`: 67 valores negativos. São erro de registro ou têm algum significado (por exemplo, ajuste ou dívida)?
2. `renda_anual`: 101 registros com "não informado" e 101 com prefixo "R$". Há diferença de origem entre esses formatos, e "não informado" significa renda desconhecida ou renda zero?
3. `data_nascimento`: 38 registros em `DD/MM/AAAA`, os demais em `AAAA-MM-DD`. Os dois formatos vêm de sistemas diferentes?
4. `ocupacao`: cerca de 31% de nulos (15.706). A ausência é legítima (por exemplo, aposentados ou sem ocupação) ou falha de preenchimento?
5. `sexo`, `possui_imovel`, `possui_carro`: pouco menos de 100 registros em cada com "Não informado". É um valor válido do cadastro ou campo não preenchido?
6. `tipo_organizacao`: 58 categorias, 45 com menos de 1% dos registros. Existe uma tabela de agrupamento oficial?

## base_submissao

7. `id_cliente`: 400 clientes aparecem duas vezes, no mesmo dia, com o mesmo tipo de contrato e a mesma parcela, mas com `valor_credito` ligeiramente diferente (mediana de 3,3%). São reenvios de solicitação ou duplicidade de registro? Qual linha vale para a decisão de crédito?
8. `valor_credito`: 63 linhas com valor cerca de 100 vezes maior que o esperado (razão crédito/bem entre 80 e 200, por exemplo 135.000.000 para um bem de 1.350.000). Há registro em centavos em alguma origem? Confirmam a correção dividindo por 100?
9. `valor_credito` / `valor_bem`: 69 linhas com crédito de 2 a 5 vezes o valor do bem e 12 com crédito abaixo de metade do bem. Isso é esperado (por exemplo, juros ou entrada) ou erro?
10. `hora_solicitacao`: 50 horas negativas (`-1:xx:xx`). Qual é a origem (fuso horário, erro de sistema) e é possível recuperar a hora real?
11. `valor_parcela`: 67 zeros e 5 nulos. Um contrato pode ter parcela zero (por exemplo, carência) ou é falha de registro?
12. `valor_bem`: cerca de 4,9% de nulos, sem concentração por tipo de contrato. Ausência de bem é legítima em algum caso?
13. `dia_semana_solicitacao`: vamos derivá-lo da data de solicitação. Alguma razão para a coluna original ter outra regra de cálculo?
14. `data_solicitacao`: 50 registros em `DD/MM/AAAA`, os demais em `AAAA-MM-DD`. Mesma origem do item 3?