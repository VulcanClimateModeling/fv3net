grammar = r"""

// top level section
?netcdf: "netcdf" symbol* bracket
group: "group" ":" symbol bracket
?bracket: "{" (dimensions | variables | data | group)* "}"

// dimensions
dimensions: "dimensions" ":" [dimension_pair*]
dimension_pair: dim "=" dim_size ";"
?dim : CNAME
?dim_size: INT
    | "UNLIMITED"

// variables
variables: "variables" ":" [variable*]
?variable: (variable_decl | variable_attr)
?variable_decl: dtype symbol [var_dims] ";"
var_dims: "(" dim("," dim)* ")"
?variable_attr: symbol ":" symbol "=" value ";"
dtype: "float"  -> float
    | "double"  -> double
    | "int" -> int
    | "char"  -> char
    | "byte"  -> byte
    | "int64" -> int64

// data
data: "data" ":" [datum*]
datum: symbol "=" list ";"
list: data_value("," data_value)*
?data_value: value | "_"

// General purpose
?symbol: CNAME
?value: SIGNED_NUMBER -> num
    | ("NaN" | "NaNf") -> nan
    | ESCAPED_STRING  -> string


%import common.CNAME
%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING
%import common.INT
%import common.WS
%ignore WS
"""
