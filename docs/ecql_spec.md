
# ECQL v1.0 — Language Specification

## 1. Назначение

ECQL (E-Corp Query Language) — учебный корпоративный DSL для получения данных из внутренних систем компании E-Corp.

Цель проекта — обучить языковую модель Qwen3-1.7B переводить запросы на русском языке в корректный ECQL.

ECQL не является SQL.

## 2. Базовые правила

Каждый запрос начинается с ключевого слова `FETCH`.

Поддерживаемые сущности:

- `[EMPLOYEES]` — сотрудники.
- `[PROJECTS]` — проекты.
- `[INVENTORY]` — товары и складские остатки.
- `[DEALS]` — сделки.

Поля указываются с префиксом `@`.

Операторы сравнения:

| Оператор | Значение |
|---|---|
| `IS` | Равно |
| `NOT` | Не равно |
| `ABOVE` | Строго больше |
| `BELOW` | Строго меньше |

Логические связки:

- `&&` — логическое И.
- `||` — логическое ИЛИ.

Форматы вывода:

- `AS JSON`
- `AS TABLE`
- `AS LIST`

Указание формата необязательно.

## 3. Грамматика ECQL v1.0

```ebnf
query       = "FETCH", entity, [where_clause], [format_clause] ;

entity      = "[EMPLOYEES]"
            | "[PROJECTS]"
            | "[INVENTORY]"
            | "[DEALS]" ;

where_clause = "WHERE", condition,
               { logical_operator, condition } ;

condition    = field, comparison_operator, value ;

field        = "@", identifier ;

comparison_operator = "IS" | "NOT" | "ABOVE" | "BELOW" ;

logical_operator = "&&" | "||" ;

format_clause = "AS", output_format ;

output_format = "JSON" | "TABLE" | "LIST" ;

value        = string_literal | number ;

string_literal = "'", { character }, "'" ;

number       = digit, { digit } ;
```

Дополнительное ограничение v1.0: в одном запросе допускается только один вид логической связки. Смешивать `&&` и `||` запрещено.

Круглые скобки и вложенные условия не поддерживаются.

## 4. Схема сущностей

Следующие поля являются учебным расширением спецификации. Они добавлены для генерации синтетического датасета.

### EMPLOYEES

| Поле | Тип |
|---|---|
| @name | string |
| @city | string |
| @department | string |
| @position | string |
| @status | string |
| @salary | number |
| @age | number |
| @experience | number |

### PROJECTS

| Поле | Тип |
|---|---|
| @name | string |
| @status | string |
| @department | string |
| @priority | string |
| @budget | number |
| @team_size | number |
| @progress | number |

### INVENTORY

| Поле | Тип |
|---|---|
| @name | string |
| @category | string |
| @status | string |
| @warehouse | string |
| @price | number |
| @quantity | number |
| @stock | number |

### DEALS

| Поле | Тип |
|---|---|
| @client | string |
| @status | string |
| @region | string |
| @manager | string |
| @amount | number |
| @discount | number |
| @probability | number |

## 5. Типизация

- Строковые значения заключаются в одинарные кавычки.
- Числовые значения записываются без кавычек.
- Операторы `ABOVE` и `BELOW` применяются только к числовым полям.
- Операторы `IS` и `NOT` применяются к строковым и числовым полям.
- Все идентификаторы полей записываются строчными буквами.
- Строковые значения сравниваются с учётом регистра.

Для v1.0 поддерживаются только неотрицательные целые числа.

Экранирование кавычек, даты, NULL и логические литералы не поддерживаются.

## 6. Примеры

### Получение всех сотрудников

```ecql
FETCH [EMPLOYEES]
```

### Фильтрация по городу

```ecql
FETCH [EMPLOYEES] WHERE @city IS 'Moscow'
```

### Условие по зарплате

```ecql
FETCH [EMPLOYEES] WHERE @salary ABOVE 150000
```

### Два условия

```ecql
FETCH [EMPLOYEES] WHERE @city IS 'Moscow' && @salary ABOVE 150000
```

### Логическое ИЛИ

```ecql
FETCH [PROJECTS] WHERE @status IS 'active' || @status IS 'planned'
```

### Формат вывода

```ecql
FETCH [DEALS] WHERE @amount ABOVE 1000000 AS JSON
```

### Три условия

```ecql
FETCH [INVENTORY] WHERE @category IS 'electronics' && @price BELOW 50000 && @quantity ABOVE 10 AS TABLE
```

## 7. Недопустимые конструкции

Следующие выражения не входят в ECQL v1.0:

```sql
SELECT * FROM employees
```

```sql
FETCH EMPLOYEES
```

```sql
FETCH [EMPLOYEES] WHERE salary > 150000
```

```sql
FETCH [EMPLOYEES] WHERE @salary >= 150000
```

```sql
FETCH [EMPLOYEES] WHERE (@city IS 'Moscow' || @city IS 'Kazan') && @salary ABOVE 150000
```

Также запрещены `JOIN`, `ORDER BY`, `GROUP BY`, `LIMIT`, `IN`, `LIKE`.

## 8. Правила датасета

Планируется сформировать 200 пар вида:

- `instruction` — постоянная инструкция перевода.
- `input` — естественный запрос на русском языке.
- `output` — эталонный запрос ECQL.

Разделение:

- Train: 160 примеров.
- Test: 40 примеров.
- Random seed: 42.

Каждый эталонный ECQL должен соответствовать грамматике и схеме полей.

В тренировочной и тестовой выборках не допускаются идентичные пары или одинаковые естественно-языковые запросы.

## 9. Ограничения

ECQL v1.0 не поддерживает:

- Выбор отдельных полей результата.
- Сортировку.
- Агрегации.
- Соединение сущностей.
- Вложенные условия.
- Арифметические выражения.
- Даты и временные интервалы.
- Операторы `>=` и `<=`.

Невыразимые запросы не включаются в основной датасет.

## 10. Критерии корректности

ECQL-запрос считается синтаксически корректным, если:

1. Начинается с `FETCH`.
2. Содержит одну допустимую сущность.
3. Использует только разрешённые поля этой сущности.
4. Содержит правильно сформированные условия.
5. Соблюдает типизацию.
6. Использует допустимые логические операторы.
7. Не смешивает `&&` и `||`.
8. Содержит только разрешённый формат вывода.
9. Не содержит посторонних SQL-конструкций.
10. Полностью соответствует грамматике ECQL v1.0.
