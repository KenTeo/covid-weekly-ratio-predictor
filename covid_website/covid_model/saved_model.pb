??)
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??(
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

: *
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_35/lstm_cell_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_35/lstm_cell_35/kernel
?
/lstm_35/lstm_cell_35/kernel/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell_35/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_35/lstm_cell_35/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*6
shared_name'%lstm_35/lstm_cell_35/recurrent_kernel
?
9lstm_35/lstm_cell_35/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_35/lstm_cell_35/recurrent_kernel*
_output_shapes
:	@?*
dtype0
?
lstm_35/lstm_cell_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_35/lstm_cell_35/bias
?
-lstm_35/lstm_cell_35/bias/Read/ReadVariableOpReadVariableOplstm_35/lstm_cell_35/bias*
_output_shapes	
:?*
dtype0
?
lstm_36/lstm_cell_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*,
shared_namelstm_36/lstm_cell_36/kernel
?
/lstm_36/lstm_cell_36/kernel/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell_36/kernel*
_output_shapes
:	@?*
dtype0
?
%lstm_36/lstm_cell_36/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*6
shared_name'%lstm_36/lstm_cell_36/recurrent_kernel
?
9lstm_36/lstm_cell_36/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_36/lstm_cell_36/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
lstm_36/lstm_cell_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_36/lstm_cell_36/bias
?
-lstm_36/lstm_cell_36/bias/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell_36/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_16/kernel/m
?
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_35/lstm_cell_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_35/lstm_cell_35/kernel/m
?
6Adam/lstm_35/lstm_cell_35/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_35/lstm_cell_35/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_35/lstm_cell_35/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m
?
@Adam/lstm_35/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_35/lstm_cell_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_35/lstm_cell_35/bias/m
?
4Adam/lstm_35/lstm_cell_35/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_35/lstm_cell_35/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_36/lstm_cell_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"Adam/lstm_36/lstm_cell_36/kernel/m
?
6Adam/lstm_36/lstm_cell_36/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_36/lstm_cell_36/kernel/m*
_output_shapes
:	@?*
dtype0
?
,Adam/lstm_36/lstm_cell_36/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*=
shared_name.,Adam/lstm_36/lstm_cell_36/recurrent_kernel/m
?
@Adam/lstm_36/lstm_cell_36/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_36/lstm_cell_36/recurrent_kernel/m*
_output_shapes
:	 ?*
dtype0
?
 Adam/lstm_36/lstm_cell_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_36/lstm_cell_36/bias/m
?
4Adam/lstm_36/lstm_cell_36/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_36/lstm_cell_36/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_16/kernel/v
?
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_35/lstm_cell_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_35/lstm_cell_35/kernel/v
?
6Adam/lstm_35/lstm_cell_35/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_35/lstm_cell_35/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_35/lstm_cell_35/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v
?
@Adam/lstm_35/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_35/lstm_cell_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_35/lstm_cell_35/bias/v
?
4Adam/lstm_35/lstm_cell_35/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_35/lstm_cell_35/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_36/lstm_cell_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"Adam/lstm_36/lstm_cell_36/kernel/v
?
6Adam/lstm_36/lstm_cell_36/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_36/lstm_cell_36/kernel/v*
_output_shapes
:	@?*
dtype0
?
,Adam/lstm_36/lstm_cell_36/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*=
shared_name.,Adam/lstm_36/lstm_cell_36/recurrent_kernel/v
?
@Adam/lstm_36/lstm_cell_36/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_36/lstm_cell_36/recurrent_kernel/v*
_output_shapes
:	 ?*
dtype0
?
 Adam/lstm_36/lstm_cell_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_36/lstm_cell_36/bias/v
?
4Adam/lstm_36/lstm_cell_36/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_36/lstm_cell_36/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratem`ma&mb'mc(md)me*mf+mgvhvi&vj'vk(vl)vm*vn+vo
8
&0
'1
(2
)3
*4
+5
6
7
8
&0
'1
(2
)3
*4
+5
6
7
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
 
?
1
state_size

&kernel
'recurrent_kernel
(bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
 

&0
'1
(2

&0
'1
(2
 
?

6states
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
?
<
state_size

)kernel
*recurrent_kernel
+bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
 

)0
*1
+2

)0
*1
+2
 
?

Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_35/lstm_cell_35/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_35/lstm_cell_35/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_35/lstm_cell_35/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_36/lstm_cell_36/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_36/lstm_cell_36/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_36/lstm_cell_36/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

Q0
 
 
 

&0
'1
(2

&0
'1
(2
 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
2	variables
3trainable_variables
4regularization_losses
 
 

0
 
 
 
 

)0
*1
+2

)0
*1
+2
 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
=	variables
>trainable_variables
?regularization_losses
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	\total
	]count
^	variables
_	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_35/lstm_cell_35/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_35/lstm_cell_35/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_35/lstm_cell_35/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_36/lstm_cell_36/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_36/lstm_cell_36/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_36/lstm_cell_36/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_35/lstm_cell_35/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_35/lstm_cell_35/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_35/lstm_cell_35/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_36/lstm_cell_36/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_36/lstm_cell_36/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_36/lstm_cell_36/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_35_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_35_inputlstm_35/lstm_cell_35/kernel%lstm_35/lstm_cell_35/recurrent_kernellstm_35/lstm_cell_35/biaslstm_36/lstm_cell_36/kernel%lstm_36/lstm_cell_36/recurrent_kernellstm_36/lstm_cell_36/biasdense_16/kerneldense_16/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_403649
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_35/lstm_cell_35/kernel/Read/ReadVariableOp9lstm_35/lstm_cell_35/recurrent_kernel/Read/ReadVariableOp-lstm_35/lstm_cell_35/bias/Read/ReadVariableOp/lstm_36/lstm_cell_36/kernel/Read/ReadVariableOp9lstm_36/lstm_cell_36/recurrent_kernel/Read/ReadVariableOp-lstm_36/lstm_cell_36/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp6Adam/lstm_35/lstm_cell_35/kernel/m/Read/ReadVariableOp@Adam/lstm_35/lstm_cell_35/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_35/lstm_cell_35/bias/m/Read/ReadVariableOp6Adam/lstm_36/lstm_cell_36/kernel/m/Read/ReadVariableOp@Adam/lstm_36/lstm_cell_36/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_36/lstm_cell_36/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp6Adam/lstm_35/lstm_cell_35/kernel/v/Read/ReadVariableOp@Adam/lstm_35/lstm_cell_35/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_35/lstm_cell_35/bias/v/Read/ReadVariableOp6Adam/lstm_36/lstm_cell_36/kernel/v/Read/ReadVariableOp@Adam/lstm_36/lstm_cell_36/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_36/lstm_cell_36/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_406292
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_35/lstm_cell_35/kernel%lstm_35/lstm_cell_35/recurrent_kernellstm_35/lstm_cell_35/biaslstm_36/lstm_cell_36/kernel%lstm_36/lstm_cell_36/recurrent_kernellstm_36/lstm_cell_36/biastotalcountAdam/dense_16/kernel/mAdam/dense_16/bias/m"Adam/lstm_35/lstm_cell_35/kernel/m,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m Adam/lstm_35/lstm_cell_35/bias/m"Adam/lstm_36/lstm_cell_36/kernel/m,Adam/lstm_36/lstm_cell_36/recurrent_kernel/m Adam/lstm_36/lstm_cell_36/bias/mAdam/dense_16/kernel/vAdam/dense_16/bias/v"Adam/lstm_35/lstm_cell_35/kernel/v,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v Adam/lstm_35/lstm_cell_35/bias/v"Adam/lstm_36/lstm_cell_36/kernel/v,Adam/lstm_36/lstm_cell_36/recurrent_kernel/v Adam/lstm_36/lstm_cell_36/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_406395??&
?
?
while_cond_404790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_404790___redundant_placeholder04
0while_while_cond_404790___redundant_placeholder14
0while_while_cond_404790___redundant_placeholder24
0while_while_cond_404790___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?8
?
while_body_404630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?8
?
while_body_405336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
.__inference_sequential_20_layer_call_fn_403464
lstm_35_input
unknown:	?
	unknown_0:	@?
	unknown_1:	?
	unknown_2:	@?
	unknown_3:	 ?
	unknown_4:	?
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_403424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_405054

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_404952*
condR
while_cond_404951*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_405980c
Plstm_35_lstm_cell_35_recurrent_kernel_regularizer_square_readvariableop_resource:	@?
identity??Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpPlstm_35_lstm_cell_35_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity9lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp
?7
?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406143

inputs
states_0
states_11
matmul_readvariableop_resource:	@?3
 matmul_1_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? N
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?	
?
.__inference_sequential_20_layer_call_fn_403691

inputs
unknown:	?
	unknown_0:	@?
	unknown_1:	?
	unknown_2:	@?
	unknown_3:	 ?
	unknown_4:	?
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_403424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?8
?
while_body_404469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?8
?
while_body_404952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_402196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_402196___redundant_placeholder04
0while_while_cond_402196___redundant_placeholder14
0while_while_cond_402196___redundant_placeholder24
0while_while_cond_402196___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_402855

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_402753*
condR
while_cond_402752*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_402868

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
while_cond_404951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_404951___redundant_placeholder04
0while_while_cond_404951___redundant_placeholder14
0while_while_cond_404951___redundant_placeholder24
0while_while_cond_404951___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_lstm_cell_35_layer_call_fn_405841

inputs
states_0
states_1
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
??
?	
!__inference__wrapped_model_401676
lstm_35_inputT
Asequential_20_lstm_35_lstm_cell_35_matmul_readvariableop_resource:	?V
Csequential_20_lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	@?Q
Bsequential_20_lstm_35_lstm_cell_35_biasadd_readvariableop_resource:	?T
Asequential_20_lstm_36_lstm_cell_36_matmul_readvariableop_resource:	@?V
Csequential_20_lstm_36_lstm_cell_36_matmul_1_readvariableop_resource:	 ?Q
Bsequential_20_lstm_36_lstm_cell_36_biasadd_readvariableop_resource:	?G
5sequential_20_dense_16_matmul_readvariableop_resource: D
6sequential_20_dense_16_biasadd_readvariableop_resource:
identity??-sequential_20/dense_16/BiasAdd/ReadVariableOp?,sequential_20/dense_16/MatMul/ReadVariableOp?9sequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp?8sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOp?:sequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp?sequential_20/lstm_35/while?9sequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp?8sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOp?:sequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp?sequential_20/lstm_36/whileX
sequential_20/lstm_35/ShapeShapelstm_35_input*
T0*
_output_shapes
:s
)sequential_20/lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_20/lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_20/lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_20/lstm_35/strided_sliceStridedSlice$sequential_20/lstm_35/Shape:output:02sequential_20/lstm_35/strided_slice/stack:output:04sequential_20/lstm_35/strided_slice/stack_1:output:04sequential_20/lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_20/lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
"sequential_20/lstm_35/zeros/packedPack,sequential_20/lstm_35/strided_slice:output:0-sequential_20/lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_20/lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_20/lstm_35/zerosFill+sequential_20/lstm_35/zeros/packed:output:0*sequential_20/lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@h
&sequential_20/lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
$sequential_20/lstm_35/zeros_1/packedPack,sequential_20/lstm_35/strided_slice:output:0/sequential_20/lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_20/lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_20/lstm_35/zeros_1Fill-sequential_20/lstm_35/zeros_1/packed:output:0,sequential_20/lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@y
$sequential_20/lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_20/lstm_35/transpose	Transposelstm_35_input-sequential_20/lstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:?????????p
sequential_20/lstm_35/Shape_1Shape#sequential_20/lstm_35/transpose:y:0*
T0*
_output_shapes
:u
+sequential_20/lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_35/strided_slice_1StridedSlice&sequential_20/lstm_35/Shape_1:output:04sequential_20/lstm_35/strided_slice_1/stack:output:06sequential_20/lstm_35/strided_slice_1/stack_1:output:06sequential_20/lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_20/lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential_20/lstm_35/TensorArrayV2TensorListReserve:sequential_20/lstm_35/TensorArrayV2/element_shape:output:0.sequential_20/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Ksequential_20/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
=sequential_20/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_20/lstm_35/transpose:y:0Tsequential_20/lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+sequential_20/lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_35/strided_slice_2StridedSlice#sequential_20/lstm_35/transpose:y:04sequential_20/lstm_35/strided_slice_2/stack:output:06sequential_20/lstm_35/strided_slice_2/stack_1:output:06sequential_20/lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
8sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpAsequential_20_lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
)sequential_20/lstm_35/lstm_cell_35/MatMulMatMul.sequential_20/lstm_35/strided_slice_2:output:0@sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
:sequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpCsequential_20_lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
+sequential_20/lstm_35/lstm_cell_35/MatMul_1MatMul$sequential_20/lstm_35/zeros:output:0Bsequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&sequential_20/lstm_35/lstm_cell_35/addAddV23sequential_20/lstm_35/lstm_cell_35/MatMul:product:05sequential_20/lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
9sequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpBsequential_20_lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*sequential_20/lstm_35/lstm_cell_35/BiasAddBiasAdd*sequential_20/lstm_35/lstm_cell_35/add:z:0Asequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????t
2sequential_20/lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential_20/lstm_35/lstm_cell_35/splitSplit;sequential_20/lstm_35/lstm_cell_35/split/split_dim:output:03sequential_20/lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split?
*sequential_20/lstm_35/lstm_cell_35/SigmoidSigmoid1sequential_20/lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_20/lstm_35/lstm_cell_35/Sigmoid_1Sigmoid1sequential_20/lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
&sequential_20/lstm_35/lstm_cell_35/mulMul0sequential_20/lstm_35/lstm_cell_35/Sigmoid_1:y:0&sequential_20/lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:?????????@?
'sequential_20/lstm_35/lstm_cell_35/ReluRelu1sequential_20/lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
(sequential_20/lstm_35/lstm_cell_35/mul_1Mul.sequential_20/lstm_35/lstm_cell_35/Sigmoid:y:05sequential_20/lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
(sequential_20/lstm_35/lstm_cell_35/add_1AddV2*sequential_20/lstm_35/lstm_cell_35/mul:z:0,sequential_20/lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
,sequential_20/lstm_35/lstm_cell_35/Sigmoid_2Sigmoid1sequential_20/lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@?
)sequential_20/lstm_35/lstm_cell_35/Relu_1Relu,sequential_20/lstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
(sequential_20/lstm_35/lstm_cell_35/mul_2Mul0sequential_20/lstm_35/lstm_cell_35/Sigmoid_2:y:07sequential_20/lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
3sequential_20/lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
%sequential_20/lstm_35/TensorArrayV2_1TensorListReserve<sequential_20/lstm_35/TensorArrayV2_1/element_shape:output:0.sequential_20/lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
sequential_20/lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_20/lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(sequential_20/lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_20/lstm_35/whileWhile1sequential_20/lstm_35/while/loop_counter:output:07sequential_20/lstm_35/while/maximum_iterations:output:0#sequential_20/lstm_35/time:output:0.sequential_20/lstm_35/TensorArrayV2_1:handle:0$sequential_20/lstm_35/zeros:output:0&sequential_20/lstm_35/zeros_1:output:0.sequential_20/lstm_35/strided_slice_1:output:0Msequential_20/lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_20_lstm_35_lstm_cell_35_matmul_readvariableop_resourceCsequential_20_lstm_35_lstm_cell_35_matmul_1_readvariableop_resourceBsequential_20_lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_20_lstm_35_while_body_401446*3
cond+R)
'sequential_20_lstm_35_while_cond_401445*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
Fsequential_20/lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
8sequential_20/lstm_35/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_20/lstm_35/while:output:3Osequential_20/lstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0~
+sequential_20/lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-sequential_20/lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_35/strided_slice_3StridedSliceAsequential_20/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:04sequential_20/lstm_35/strided_slice_3/stack:output:06sequential_20/lstm_35/strided_slice_3/stack_1:output:06sequential_20/lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask{
&sequential_20/lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!sequential_20/lstm_35/transpose_1	TransposeAsequential_20/lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_20/lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@q
sequential_20/lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    p
sequential_20/lstm_36/ShapeShape%sequential_20/lstm_35/transpose_1:y:0*
T0*
_output_shapes
:s
)sequential_20/lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_20/lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_20/lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_20/lstm_36/strided_sliceStridedSlice$sequential_20/lstm_36/Shape:output:02sequential_20/lstm_36/strided_slice/stack:output:04sequential_20/lstm_36/strided_slice/stack_1:output:04sequential_20/lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_20/lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
"sequential_20/lstm_36/zeros/packedPack,sequential_20/lstm_36/strided_slice:output:0-sequential_20/lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_20/lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_20/lstm_36/zerosFill+sequential_20/lstm_36/zeros/packed:output:0*sequential_20/lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:????????? h
&sequential_20/lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
$sequential_20/lstm_36/zeros_1/packedPack,sequential_20/lstm_36/strided_slice:output:0/sequential_20/lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_20/lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_20/lstm_36/zeros_1Fill-sequential_20/lstm_36/zeros_1/packed:output:0,sequential_20/lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? y
$sequential_20/lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_20/lstm_36/transpose	Transpose%sequential_20/lstm_35/transpose_1:y:0-sequential_20/lstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@p
sequential_20/lstm_36/Shape_1Shape#sequential_20/lstm_36/transpose:y:0*
T0*
_output_shapes
:u
+sequential_20/lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_36/strided_slice_1StridedSlice&sequential_20/lstm_36/Shape_1:output:04sequential_20/lstm_36/strided_slice_1/stack:output:06sequential_20/lstm_36/strided_slice_1/stack_1:output:06sequential_20/lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_20/lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential_20/lstm_36/TensorArrayV2TensorListReserve:sequential_20/lstm_36/TensorArrayV2/element_shape:output:0.sequential_20/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Ksequential_20/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=sequential_20/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_20/lstm_36/transpose:y:0Tsequential_20/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???u
+sequential_20/lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_36/strided_slice_2StridedSlice#sequential_20/lstm_36/transpose:y:04sequential_20/lstm_36/strided_slice_2/stack:output:06sequential_20/lstm_36/strided_slice_2/stack_1:output:06sequential_20/lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
8sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOpReadVariableOpAsequential_20_lstm_36_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
)sequential_20/lstm_36/lstm_cell_36/MatMulMatMul.sequential_20/lstm_36/strided_slice_2:output:0@sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
:sequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOpCsequential_20_lstm_36_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
+sequential_20/lstm_36/lstm_cell_36/MatMul_1MatMul$sequential_20/lstm_36/zeros:output:0Bsequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&sequential_20/lstm_36/lstm_cell_36/addAddV23sequential_20/lstm_36/lstm_cell_36/MatMul:product:05sequential_20/lstm_36/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
9sequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOpBsequential_20_lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*sequential_20/lstm_36/lstm_cell_36/BiasAddBiasAdd*sequential_20/lstm_36/lstm_cell_36/add:z:0Asequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????t
2sequential_20/lstm_36/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential_20/lstm_36/lstm_cell_36/splitSplit;sequential_20/lstm_36/lstm_cell_36/split/split_dim:output:03sequential_20/lstm_36/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split?
*sequential_20/lstm_36/lstm_cell_36/SigmoidSigmoid1sequential_20/lstm_36/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
,sequential_20/lstm_36/lstm_cell_36/Sigmoid_1Sigmoid1sequential_20/lstm_36/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
&sequential_20/lstm_36/lstm_cell_36/mulMul0sequential_20/lstm_36/lstm_cell_36/Sigmoid_1:y:0&sequential_20/lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:????????? ?
'sequential_20/lstm_36/lstm_cell_36/ReluRelu1sequential_20/lstm_36/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
(sequential_20/lstm_36/lstm_cell_36/mul_1Mul.sequential_20/lstm_36/lstm_cell_36/Sigmoid:y:05sequential_20/lstm_36/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
(sequential_20/lstm_36/lstm_cell_36/add_1AddV2*sequential_20/lstm_36/lstm_cell_36/mul:z:0,sequential_20/lstm_36/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
,sequential_20/lstm_36/lstm_cell_36/Sigmoid_2Sigmoid1sequential_20/lstm_36/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? ?
)sequential_20/lstm_36/lstm_cell_36/Relu_1Relu,sequential_20/lstm_36/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
(sequential_20/lstm_36/lstm_cell_36/mul_2Mul0sequential_20/lstm_36/lstm_cell_36/Sigmoid_2:y:07sequential_20/lstm_36/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
3sequential_20/lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
%sequential_20/lstm_36/TensorArrayV2_1TensorListReserve<sequential_20/lstm_36/TensorArrayV2_1/element_shape:output:0.sequential_20/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???\
sequential_20/lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_20/lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????j
(sequential_20/lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_20/lstm_36/whileWhile1sequential_20/lstm_36/while/loop_counter:output:07sequential_20/lstm_36/while/maximum_iterations:output:0#sequential_20/lstm_36/time:output:0.sequential_20/lstm_36/TensorArrayV2_1:handle:0$sequential_20/lstm_36/zeros:output:0&sequential_20/lstm_36/zeros_1:output:0.sequential_20/lstm_36/strided_slice_1:output:0Msequential_20/lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_20_lstm_36_lstm_cell_36_matmul_readvariableop_resourceCsequential_20_lstm_36_lstm_cell_36_matmul_1_readvariableop_resourceBsequential_20_lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_20_lstm_36_while_body_401585*3
cond+R)
'sequential_20_lstm_36_while_cond_401584*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
Fsequential_20/lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
8sequential_20/lstm_36/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_20/lstm_36/while:output:3Osequential_20/lstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0~
+sequential_20/lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????w
-sequential_20/lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_20/lstm_36/strided_slice_3StridedSliceAsequential_20/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:04sequential_20/lstm_36/strided_slice_3/stack:output:06sequential_20/lstm_36/strided_slice_3/stack_1:output:06sequential_20/lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask{
&sequential_20/lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!sequential_20/lstm_36/transpose_1	TransposeAsequential_20/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_20/lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? q
sequential_20/lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
!sequential_20/dropout_16/IdentityIdentity.sequential_20/lstm_36/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? ?
,sequential_20/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
sequential_20/dense_16/MatMulMatMul*sequential_20/dropout_16/Identity:output:04sequential_20/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_20/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_20/dense_16/BiasAddBiasAdd'sequential_20/dense_16/MatMul:product:05sequential_20/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_20/dense_16/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^sequential_20/dense_16/BiasAdd/ReadVariableOp-^sequential_20/dense_16/MatMul/ReadVariableOp:^sequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp9^sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOp;^sequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp^sequential_20/lstm_35/while:^sequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp9^sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOp;^sequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp^sequential_20/lstm_36/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2^
-sequential_20/dense_16/BiasAdd/ReadVariableOp-sequential_20/dense_16/BiasAdd/ReadVariableOp2\
,sequential_20/dense_16/MatMul/ReadVariableOp,sequential_20/dense_16/MatMul/ReadVariableOp2v
9sequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp9sequential_20/lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2t
8sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOp8sequential_20/lstm_35/lstm_cell_35/MatMul/ReadVariableOp2x
:sequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:sequential_20/lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2:
sequential_20/lstm_35/whilesequential_20/lstm_35/while2v
9sequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp9sequential_20/lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp2t
8sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOp8sequential_20/lstm_36/lstm_cell_36/MatMul/ReadVariableOp2x
:sequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp:sequential_20/lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp2:
sequential_20/lstm_36/whilesequential_20/lstm_36/while:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404893

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_404791*
condR
while_cond_404790*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?O
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_402923

inputs!
lstm_35_402688:	?!
lstm_35_402690:	@?
lstm_35_402692:	?!
lstm_36_402856:	@?!
lstm_36_402858:	 ?
lstm_36_402860:	?!
dense_16_402881: 
dense_16_402883:
identity?? dense_16/StatefulPartitionedCall?lstm_35/StatefulPartitionedCall?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/StatefulPartitionedCall?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?
lstm_35/StatefulPartitionedCallStatefulPartitionedCallinputslstm_35_402688lstm_35_402690lstm_35_402692*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_402687?
lstm_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0lstm_36_402856lstm_36_402858lstm_36_402860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_402855?
dropout_16/PartitionedCallPartitionedCall(lstm_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402868?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_402881dense_16_402883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_402880?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_402688*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_402690*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_402692*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_402856*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_402858*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_402860*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp ^lstm_36/StatefulPartitionedCall<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406093

inputs
states_0
states_11
matmul_readvariableop_resource:	@?3
 matmul_1_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? N
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?8
?
while_body_403230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404571
inputs_0>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_404469*
condR
while_cond_404468*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?7
?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401761

inputs

states
states_11
matmul_readvariableop_resource:	?3
 matmul_1_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?
?
__inference_loss_fn_3_406154Y
Flstm_36_lstm_cell_36_kernel_regularizer_square_readvariableop_resource:	@?
identity??=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_36_lstm_cell_36_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/lstm_36/lstm_cell_36/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp
?
?
(__inference_lstm_35_layer_call_fn_404377
inputs_0
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_401862|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?"
?
while_body_402197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_36_402221_0:	@?.
while_lstm_cell_36_402223_0:	 ?*
while_lstm_cell_36_402225_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_36_402221:	@?,
while_lstm_cell_36_402223:	 ?(
while_lstm_cell_36_402225:	???*while/lstm_cell_36/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
*while/lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_36_402221_0while_lstm_cell_36_402223_0while_lstm_cell_36_402225_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402183?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity3while/lstm_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:????????? ?
while/Identity_5Identity3while/lstm_cell_36/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:????????? y

while/NoOpNoOp+^while/lstm_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_36_402221while_lstm_cell_36_402221_0"8
while_lstm_cell_36_402223while_lstm_cell_36_402223_0"8
while_lstm_cell_36_402225while_lstm_cell_36_402225_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2X
*while/lstm_cell_36/StatefulPartitionedCall*while/lstm_cell_36/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_lstm_36_layer_call_fn_405116

inputs
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_403149o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_405787

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?7
?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402183

inputs

states
states_11
matmul_readvariableop_resource:	@?3
 matmul_1_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? N
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?O
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_403524
lstm_35_input!
lstm_35_403467:	?!
lstm_35_403469:	@?
lstm_35_403471:	?!
lstm_36_403474:	@?!
lstm_36_403476:	 ?
lstm_36_403478:	?!
dense_16_403482: 
dense_16_403484:
identity?? dense_16/StatefulPartitionedCall?lstm_35/StatefulPartitionedCall?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/StatefulPartitionedCall?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?
lstm_35/StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputlstm_35_403467lstm_35_403469lstm_35_403471*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_402687?
lstm_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0lstm_36_403474lstm_36_403476lstm_36_403478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_402855?
dropout_16/PartitionedCallPartitionedCall(lstm_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402868?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_403482dense_16_403484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_402880?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403467*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403469*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403471*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403474*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403476*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403478*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp ^lstm_36/StatefulPartitionedCall<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405760

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_405658*
condR
while_cond_405657*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_403332

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_403230*
condR
while_cond_403229*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405908

inputs
states_0
states_11
matmul_readvariableop_resource:	?3
 matmul_1_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?7
?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401943

inputs

states
states_11
matmul_readvariableop_resource:	?3
 matmul_1_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?
?
while_cond_403229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_403229___redundant_placeholder04
0while_while_cond_403229___redundant_placeholder14
0while_while_cond_403229___redundant_placeholder24
0while_while_cond_403229___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405277
inputs_0>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_405175*
condR
while_cond_405174*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?U
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_402089

inputs&
lstm_cell_35_401989:	?&
lstm_cell_35_401991:	@?"
lstm_cell_35_401993:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_35/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_401989lstm_cell_35_401991lstm_cell_35_401993*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401943n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_401989lstm_cell_35_401991lstm_cell_35_401993*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_402002*
condR
while_cond_402001*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401989*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401991*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401993*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_403046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_403046___redundant_placeholder04
0while_while_cond_403046___redundant_placeholder14
0while_while_cond_403046___redundant_placeholder24
0while_while_cond_403046___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
??
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_404348

inputsF
3lstm_35_lstm_cell_35_matmul_readvariableop_resource:	?H
5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	@?C
4lstm_35_lstm_cell_35_biasadd_readvariableop_resource:	?F
3lstm_36_lstm_cell_36_matmul_readvariableop_resource:	@?H
5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource:	 ?C
4lstm_36_lstm_cell_36_biasadd_readvariableop_resource:	?9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp?*lstm_35/lstm_cell_35/MatMul/ReadVariableOp?,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_35/while?+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp?*lstm_36/lstm_cell_36/MatMul/ReadVariableOp?,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/whileC
lstm_35/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_sliceStridedSlicelstm_35/Shape:output:0$lstm_35/strided_slice/stack:output:0&lstm_35/strided_slice/stack_1:output:0&lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
lstm_35/zeros/packedPacklstm_35/strided_slice:output:0lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_35/zerosFilllstm_35/zeros/packed:output:0lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@Z
lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
lstm_35/zeros_1/packedPacklstm_35/strided_slice:output:0!lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_35/zeros_1Filllstm_35/zeros_1/packed:output:0lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@k
lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_35/transpose	Transposeinputslstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:?????????T
lstm_35/Shape_1Shapelstm_35/transpose:y:0*
T0*
_output_shapes
:g
lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_1StridedSlicelstm_35/Shape_1:output:0&lstm_35/strided_slice_1/stack:output:0(lstm_35/strided_slice_1/stack_1:output:0(lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_35/TensorArrayV2TensorListReserve,lstm_35/TensorArrayV2/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_35/transpose:y:0Flstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???g
lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_2StridedSlicelstm_35/transpose:y:0&lstm_35/strided_slice_2/stack:output:0(lstm_35/strided_slice_2/stack_1:output:0(lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
*lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_35/lstm_cell_35/MatMulMatMul lstm_35/strided_slice_2:output:02lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_35/lstm_cell_35/MatMul_1MatMullstm_35/zeros:output:04lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_35/lstm_cell_35/addAddV2%lstm_35/lstm_cell_35/MatMul:product:0'lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_35/lstm_cell_35/BiasAddBiasAddlstm_35/lstm_cell_35/add:z:03lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
$lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_35/lstm_cell_35/splitSplit-lstm_35/lstm_cell_35/split/split_dim:output:0%lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split~
lstm_35/lstm_cell_35/SigmoidSigmoid#lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/Sigmoid_1Sigmoid#lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mulMul"lstm_35/lstm_cell_35/Sigmoid_1:y:0lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:?????????@x
lstm_35/lstm_cell_35/ReluRelu#lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mul_1Mul lstm_35/lstm_cell_35/Sigmoid:y:0'lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/add_1AddV2lstm_35/lstm_cell_35/mul:z:0lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/Sigmoid_2Sigmoid#lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@u
lstm_35/lstm_cell_35/Relu_1Relulstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mul_2Mul"lstm_35/lstm_cell_35/Sigmoid_2:y:0)lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@v
%lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
lstm_35/TensorArrayV2_1TensorListReserve.lstm_35/TensorArrayV2_1/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???N
lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????\
lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_35/whileWhile#lstm_35/while/loop_counter:output:0)lstm_35/while/maximum_iterations:output:0lstm_35/time:output:0 lstm_35/TensorArrayV2_1:handle:0lstm_35/zeros:output:0lstm_35/zeros_1:output:0 lstm_35/strided_slice_1:output:0?lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_35_lstm_cell_35_matmul_readvariableop_resource5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_35_while_body_404075*%
condR
lstm_35_while_cond_404074*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
*lstm_35/TensorArrayV2Stack/TensorListStackTensorListStacklstm_35/while:output:3Alstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0p
lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????i
lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_3StridedSlice3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_35/strided_slice_3/stack:output:0(lstm_35/strided_slice_3/stack_1:output:0(lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskm
lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_35/transpose_1	Transpose3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@c
lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_36/ShapeShapelstm_35/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_sliceStridedSlicelstm_36/Shape:output:0$lstm_36/strided_slice/stack:output:0&lstm_36/strided_slice/stack_1:output:0&lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/zeros/packedPacklstm_36/strided_slice:output:0lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_36/zerosFilllstm_36/zeros/packed:output:0lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:????????? Z
lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/zeros_1/packedPacklstm_36/strided_slice:output:0!lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_36/zeros_1Filllstm_36/zeros_1/packed:output:0lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? k
lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_36/transpose	Transposelstm_35/transpose_1:y:0lstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@T
lstm_36/Shape_1Shapelstm_36/transpose:y:0*
T0*
_output_shapes
:g
lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_1StridedSlicelstm_36/Shape_1:output:0&lstm_36/strided_slice_1/stack:output:0(lstm_36/strided_slice_1/stack_1:output:0(lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_36/TensorArrayV2TensorListReserve,lstm_36/TensorArrayV2/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_36/transpose:y:0Flstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???g
lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_2StridedSlicelstm_36/transpose:y:0&lstm_36/strided_slice_2/stack:output:0(lstm_36/strided_slice_2/stack_1:output:0(lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*lstm_36/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_36/lstm_cell_36/MatMulMatMul lstm_36/strided_slice_2:output:02lstm_36/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_36/lstm_cell_36/MatMul_1MatMullstm_36/zeros:output:04lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_36/lstm_cell_36/addAddV2%lstm_36/lstm_cell_36/MatMul:product:0'lstm_36/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_36/lstm_cell_36/BiasAddBiasAddlstm_36/lstm_cell_36/add:z:03lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
$lstm_36/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_36/lstm_cell_36/splitSplit-lstm_36/lstm_cell_36/split/split_dim:output:0%lstm_36/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split~
lstm_36/lstm_cell_36/SigmoidSigmoid#lstm_36/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/Sigmoid_1Sigmoid#lstm_36/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mulMul"lstm_36/lstm_cell_36/Sigmoid_1:y:0lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:????????? x
lstm_36/lstm_cell_36/ReluRelu#lstm_36/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mul_1Mul lstm_36/lstm_cell_36/Sigmoid:y:0'lstm_36/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/add_1AddV2lstm_36/lstm_cell_36/mul:z:0lstm_36/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/Sigmoid_2Sigmoid#lstm_36/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? u
lstm_36/lstm_cell_36/Relu_1Relulstm_36/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mul_2Mul"lstm_36/lstm_cell_36/Sigmoid_2:y:0)lstm_36/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? v
%lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
lstm_36/TensorArrayV2_1TensorListReserve.lstm_36/TensorArrayV2_1/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???N
lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????\
lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/whileWhile#lstm_36/while/loop_counter:output:0)lstm_36/while/maximum_iterations:output:0lstm_36/time:output:0 lstm_36/TensorArrayV2_1:handle:0lstm_36/zeros:output:0lstm_36/zeros_1:output:0 lstm_36/strided_slice_1:output:0?lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_36_lstm_cell_36_matmul_readvariableop_resource5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_36_while_body_404214*%
condR
lstm_36_while_cond_404213*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
*lstm_36/TensorArrayV2Stack/TensorListStackTensorListStacklstm_36/while:output:3Alstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0p
lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????i
lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_3StridedSlice3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_36/strided_slice_3/stack:output:0(lstm_36/strided_slice_3/stack_1:output:0(lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskm
lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_36/transpose_1	Transpose3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? c
lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_16/dropout/MulMul lstm_36/strided_slice_3:output:0!dropout_16/dropout/Const:output:0*
T0*'
_output_shapes
:????????? h
dropout_16/dropout/ShapeShape lstm_36/strided_slice_3:output:0*
T0*
_output_shapes
:?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_16/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_16/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp,^lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+^lstm_35/lstm_cell_35/MatMul/ReadVariableOp-^lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm_35/while,^lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp+^lstm_36/lstm_cell_36/MatMul/ReadVariableOp-^lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm_36/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2Z
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2X
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp*lstm_35/lstm_cell_35/MatMul/ReadVariableOp2\
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2
lstm_35/whilelstm_35/while2Z
+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp2X
*lstm_36/lstm_cell_36/MatMul/ReadVariableOp*lstm_36/lstm_cell_36/MatMul/ReadVariableOp2\
,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2
lstm_36/whilelstm_36/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_404629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_404629___redundant_placeholder04
0while_while_cond_404629___redundant_placeholder14
0while_while_cond_404629___redundant_placeholder24
0while_while_cond_404629___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
__inference_loss_fn_4_406165c
Plstm_36_lstm_cell_36_recurrent_kernel_regularizer_square_readvariableop_resource:	 ?
identity??Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpPlstm_36_lstm_cell_36_recurrent_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity9lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp
?
?
(__inference_lstm_35_layer_call_fn_404410

inputs
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_403332s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_403584
lstm_35_input!
lstm_35_403527:	?!
lstm_35_403529:	@?
lstm_35_403531:	?!
lstm_36_403534:	@?!
lstm_36_403536:	 ?
lstm_36_403538:	?!
dense_16_403542: 
dense_16_403544:
identity?? dense_16/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?lstm_35/StatefulPartitionedCall?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/StatefulPartitionedCall?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?
lstm_35/StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputlstm_35_403527lstm_35_403529lstm_35_403531*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_403332?
lstm_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0lstm_36_403534lstm_36_403536lstm_36_403538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_403149?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402972?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_403542dense_16_403544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_402880?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403527*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403529*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403531*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403534*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403536*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403538*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp ^lstm_36/StatefulPartitionedCall<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?
?
'sequential_20_lstm_35_while_cond_401445H
Dsequential_20_lstm_35_while_sequential_20_lstm_35_while_loop_counterN
Jsequential_20_lstm_35_while_sequential_20_lstm_35_while_maximum_iterations+
'sequential_20_lstm_35_while_placeholder-
)sequential_20_lstm_35_while_placeholder_1-
)sequential_20_lstm_35_while_placeholder_2-
)sequential_20_lstm_35_while_placeholder_3J
Fsequential_20_lstm_35_while_less_sequential_20_lstm_35_strided_slice_1`
\sequential_20_lstm_35_while_sequential_20_lstm_35_while_cond_401445___redundant_placeholder0`
\sequential_20_lstm_35_while_sequential_20_lstm_35_while_cond_401445___redundant_placeholder1`
\sequential_20_lstm_35_while_sequential_20_lstm_35_while_cond_401445___redundant_placeholder2`
\sequential_20_lstm_35_while_sequential_20_lstm_35_while_cond_401445___redundant_placeholder3(
$sequential_20_lstm_35_while_identity
?
 sequential_20/lstm_35/while/LessLess'sequential_20_lstm_35_while_placeholderFsequential_20_lstm_35_while_less_sequential_20_lstm_35_strided_slice_1*
T0*
_output_shapes
: w
$sequential_20/lstm_35/while/IdentityIdentity$sequential_20/lstm_35/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_20_lstm_35_while_identity-sequential_20/lstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?U
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_402511

inputs&
lstm_cell_36_402411:	@?&
lstm_cell_36_402413:	 ?"
lstm_cell_36_402415:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_36/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
$lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_36_402411lstm_cell_36_402413lstm_cell_36_402415*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402365n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_36_402411lstm_cell_36_402413lstm_cell_36_402415*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_402424*
condR
while_cond_402423*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402411*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402413*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402415*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_36/StatefulPartitionedCall$lstm_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?A
?

lstm_36_while_body_403889,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3+
'lstm_36_while_lstm_36_strided_slice_1_0g
clstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0:	@?P
=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?K
<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
lstm_36_while_identity
lstm_36_while_identity_1
lstm_36_while_identity_2
lstm_36_while_identity_3
lstm_36_while_identity_4
lstm_36_while_identity_5)
%lstm_36_while_lstm_36_strided_slice_1e
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorL
9lstm_36_while_lstm_cell_36_matmul_readvariableop_resource:	@?N
;lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?I
:lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource:	???1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp?0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp?2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp?
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
1lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0lstm_36_while_placeholderHlstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
!lstm_36/while/lstm_cell_36/MatMulMatMul8lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
#lstm_36/while/lstm_cell_36/MatMul_1MatMullstm_36_while_placeholder_2:lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_36/while/lstm_cell_36/addAddV2+lstm_36/while/lstm_cell_36/MatMul:product:0-lstm_36/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
"lstm_36/while/lstm_cell_36/BiasAddBiasAdd"lstm_36/while/lstm_cell_36/add:z:09lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????l
*lstm_36/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_36/while/lstm_cell_36/splitSplit3lstm_36/while/lstm_cell_36/split/split_dim:output:0+lstm_36/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split?
"lstm_36/while/lstm_cell_36/SigmoidSigmoid)lstm_36/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
$lstm_36/while/lstm_cell_36/Sigmoid_1Sigmoid)lstm_36/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
lstm_36/while/lstm_cell_36/mulMul(lstm_36/while/lstm_cell_36/Sigmoid_1:y:0lstm_36_while_placeholder_3*
T0*'
_output_shapes
:????????? ?
lstm_36/while/lstm_cell_36/ReluRelu)lstm_36/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/mul_1Mul&lstm_36/while/lstm_cell_36/Sigmoid:y:0-lstm_36/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/add_1AddV2"lstm_36/while/lstm_cell_36/mul:z:0$lstm_36/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
$lstm_36/while/lstm_cell_36/Sigmoid_2Sigmoid)lstm_36/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? ?
!lstm_36/while/lstm_cell_36/Relu_1Relu$lstm_36/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/mul_2Mul(lstm_36/while/lstm_cell_36/Sigmoid_2:y:0/lstm_36/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
2lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_36_while_placeholder_1lstm_36_while_placeholder$lstm_36/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???U
lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_36/while/addAddV2lstm_36_while_placeholderlstm_36/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_36/while/add_1AddV2(lstm_36_while_lstm_36_while_loop_counterlstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_36/while/IdentityIdentitylstm_36/while/add_1:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_1Identity.lstm_36_while_lstm_36_while_maximum_iterations^lstm_36/while/NoOp*
T0*
_output_shapes
: q
lstm_36/while/Identity_2Identitylstm_36/while/add:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_3IdentityBlstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_4Identity$lstm_36/while/lstm_cell_36/mul_2:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
lstm_36/while/Identity_5Identity$lstm_36/while/lstm_cell_36/add_1:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
lstm_36/while/NoOpNoOp2^lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp1^lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp3^lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_36_while_identitylstm_36/while/Identity:output:0"=
lstm_36_while_identity_1!lstm_36/while/Identity_1:output:0"=
lstm_36_while_identity_2!lstm_36/while/Identity_2:output:0"=
lstm_36_while_identity_3!lstm_36/while/Identity_3:output:0"=
lstm_36_while_identity_4!lstm_36/while/Identity_4:output:0"=
lstm_36_while_identity_5!lstm_36/while/Identity_5:output:0"P
%lstm_36_while_lstm_36_strided_slice_1'lstm_36_while_lstm_36_strided_slice_1_0"z
:lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0"|
;lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0"x
9lstm_36_while_lstm_cell_36_matmul_readvariableop_resource;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0"?
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2f
1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp2d
0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp2h
2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
$__inference_signature_wrapper_403649
lstm_35_input
unknown:	?
	unknown_0:	@?
	unknown_1:	?
	unknown_2:	@?
	unknown_3:	 ?
	unknown_4:	?
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_401676o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?8
?
while_body_402585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?7
?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405958

inputs
states_0
states_11
matmul_readvariableop_resource:	?3
 matmul_1_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????@Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405599

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_405497*
condR
while_cond_405496*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_lstm_35_layer_call_fn_404388
inputs_0
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_402089|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_402752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_402752___redundant_placeholder04
0while_while_cond_402752___redundant_placeholder14
0while_while_cond_402752___redundant_placeholder24
0while_while_cond_402752___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?

?
lstm_36_while_cond_403888,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3.
*lstm_36_while_less_lstm_36_strided_slice_1D
@lstm_36_while_lstm_36_while_cond_403888___redundant_placeholder0D
@lstm_36_while_lstm_36_while_cond_403888___redundant_placeholder1D
@lstm_36_while_lstm_36_while_cond_403888___redundant_placeholder2D
@lstm_36_while_lstm_36_while_cond_403888___redundant_placeholder3
lstm_36_while_identity
?
lstm_36/while/LessLesslstm_36_while_placeholder*lstm_36_while_less_lstm_36_strided_slice_1*
T0*
_output_shapes
: [
lstm_36/while/IdentityIdentitylstm_36/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_36_while_identitylstm_36/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?F
?
__inference__traced_save_406292
file_prefix.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_35_lstm_cell_35_kernel_read_readvariableopD
@savev2_lstm_35_lstm_cell_35_recurrent_kernel_read_readvariableop8
4savev2_lstm_35_lstm_cell_35_bias_read_readvariableop:
6savev2_lstm_36_lstm_cell_36_kernel_read_readvariableopD
@savev2_lstm_36_lstm_cell_36_recurrent_kernel_read_readvariableop8
4savev2_lstm_36_lstm_cell_36_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableopA
=savev2_adam_lstm_35_lstm_cell_35_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_35_lstm_cell_35_bias_m_read_readvariableopA
=savev2_adam_lstm_36_lstm_cell_36_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_36_lstm_cell_36_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_36_lstm_cell_36_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableopA
=savev2_adam_lstm_35_lstm_cell_35_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_35_lstm_cell_35_bias_v_read_readvariableopA
=savev2_adam_lstm_36_lstm_cell_36_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_36_lstm_cell_36_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_36_lstm_cell_36_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_35_lstm_cell_35_kernel_read_readvariableop@savev2_lstm_35_lstm_cell_35_recurrent_kernel_read_readvariableop4savev2_lstm_35_lstm_cell_35_bias_read_readvariableop6savev2_lstm_36_lstm_cell_36_kernel_read_readvariableop@savev2_lstm_36_lstm_cell_36_recurrent_kernel_read_readvariableop4savev2_lstm_36_lstm_cell_36_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop=savev2_adam_lstm_35_lstm_cell_35_kernel_m_read_readvariableopGsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_35_lstm_cell_35_bias_m_read_readvariableop=savev2_adam_lstm_36_lstm_cell_36_kernel_m_read_readvariableopGsavev2_adam_lstm_36_lstm_cell_36_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_36_lstm_cell_36_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop=savev2_adam_lstm_35_lstm_cell_35_kernel_v_read_readvariableopGsavev2_adam_lstm_35_lstm_cell_35_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_35_lstm_cell_35_bias_v_read_readvariableop=savev2_adam_lstm_36_lstm_cell_36_kernel_v_read_readvariableopGsavev2_adam_lstm_36_lstm_cell_36_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_36_lstm_cell_36_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : :: : : : : :	?:	@?:?:	@?:	 ?:?: : : ::	?:	@?:?:	@?:	 ?:?: ::	?:	@?:?:	@?:	 ?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%	!

_output_shapes
:	@?:!


_output_shapes	
:?:%!

_output_shapes
:	@?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	@?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	@?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?: 

_output_shapes
: 
?
?
'sequential_20_lstm_36_while_cond_401584H
Dsequential_20_lstm_36_while_sequential_20_lstm_36_while_loop_counterN
Jsequential_20_lstm_36_while_sequential_20_lstm_36_while_maximum_iterations+
'sequential_20_lstm_36_while_placeholder-
)sequential_20_lstm_36_while_placeholder_1-
)sequential_20_lstm_36_while_placeholder_2-
)sequential_20_lstm_36_while_placeholder_3J
Fsequential_20_lstm_36_while_less_sequential_20_lstm_36_strided_slice_1`
\sequential_20_lstm_36_while_sequential_20_lstm_36_while_cond_401584___redundant_placeholder0`
\sequential_20_lstm_36_while_sequential_20_lstm_36_while_cond_401584___redundant_placeholder1`
\sequential_20_lstm_36_while_sequential_20_lstm_36_while_cond_401584___redundant_placeholder2`
\sequential_20_lstm_36_while_sequential_20_lstm_36_while_cond_401584___redundant_placeholder3(
$sequential_20_lstm_36_while_identity
?
 sequential_20/lstm_36/while/LessLess'sequential_20_lstm_36_while_placeholderFsequential_20_lstm_36_while_less_sequential_20_lstm_36_strided_slice_1*
T0*
_output_shapes
: w
$sequential_20/lstm_36/while/IdentityIdentity$sequential_20/lstm_36/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_20_lstm_36_while_identity-sequential_20/lstm_36/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?"
?
while_body_402002
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_35_402026_0:	?.
while_lstm_cell_35_402028_0:	@?*
while_lstm_cell_35_402030_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_35_402026:	?,
while_lstm_cell_35_402028:	@?(
while_lstm_cell_35_402030:	???*while/lstm_cell_35/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_402026_0while_lstm_cell_35_402028_0while_lstm_cell_35_402030_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401943?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@?
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????@y

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_35_402026while_lstm_cell_35_402026_0"8
while_lstm_cell_35_402028while_lstm_cell_35_402028_0"8
while_lstm_cell_35_402030while_lstm_cell_35_402030_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?R
?
'sequential_20_lstm_35_while_body_401446H
Dsequential_20_lstm_35_while_sequential_20_lstm_35_while_loop_counterN
Jsequential_20_lstm_35_while_sequential_20_lstm_35_while_maximum_iterations+
'sequential_20_lstm_35_while_placeholder-
)sequential_20_lstm_35_while_placeholder_1-
)sequential_20_lstm_35_while_placeholder_2-
)sequential_20_lstm_35_while_placeholder_3G
Csequential_20_lstm_35_while_sequential_20_lstm_35_strided_slice_1_0?
sequential_20_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_35_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_20_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	?^
Ksequential_20_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?Y
Jsequential_20_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:	?(
$sequential_20_lstm_35_while_identity*
&sequential_20_lstm_35_while_identity_1*
&sequential_20_lstm_35_while_identity_2*
&sequential_20_lstm_35_while_identity_3*
&sequential_20_lstm_35_while_identity_4*
&sequential_20_lstm_35_while_identity_5E
Asequential_20_lstm_35_while_sequential_20_lstm_35_strided_slice_1?
}sequential_20_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_35_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_20_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	?\
Isequential_20_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	@?W
Hsequential_20_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:	????sequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?>sequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp?@sequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp?
Msequential_20/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?sequential_20/lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_20_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_35_tensorarrayunstack_tensorlistfromtensor_0'sequential_20_lstm_35_while_placeholderVsequential_20/lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
>sequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOpIsequential_20_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
/sequential_20/lstm_35/while/lstm_cell_35/MatMulMatMulFsequential_20/lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
@sequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOpKsequential_20_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
1sequential_20/lstm_35/while/lstm_cell_35/MatMul_1MatMul)sequential_20_lstm_35_while_placeholder_2Hsequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_20/lstm_35/while/lstm_cell_35/addAddV29sequential_20/lstm_35/while/lstm_cell_35/MatMul:product:0;sequential_20/lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
?sequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOpJsequential_20_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
0sequential_20/lstm_35/while/lstm_cell_35/BiasAddBiasAdd0sequential_20/lstm_35/while/lstm_cell_35/add:z:0Gsequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
8sequential_20/lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
.sequential_20/lstm_35/while/lstm_cell_35/splitSplitAsequential_20/lstm_35/while/lstm_cell_35/split/split_dim:output:09sequential_20/lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split?
0sequential_20/lstm_35/while/lstm_cell_35/SigmoidSigmoid7sequential_20/lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
2sequential_20/lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid7sequential_20/lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
,sequential_20/lstm_35/while/lstm_cell_35/mulMul6sequential_20/lstm_35/while/lstm_cell_35/Sigmoid_1:y:0)sequential_20_lstm_35_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
-sequential_20/lstm_35/while/lstm_cell_35/ReluRelu7sequential_20/lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
.sequential_20/lstm_35/while/lstm_cell_35/mul_1Mul4sequential_20/lstm_35/while/lstm_cell_35/Sigmoid:y:0;sequential_20/lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
.sequential_20/lstm_35/while/lstm_cell_35/add_1AddV20sequential_20/lstm_35/while/lstm_cell_35/mul:z:02sequential_20/lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
2sequential_20/lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid7sequential_20/lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@?
/sequential_20/lstm_35/while/lstm_cell_35/Relu_1Relu2sequential_20/lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
.sequential_20/lstm_35/while/lstm_cell_35/mul_2Mul6sequential_20/lstm_35/while/lstm_cell_35/Sigmoid_2:y:0=sequential_20/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
@sequential_20/lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_20_lstm_35_while_placeholder_1'sequential_20_lstm_35_while_placeholder2sequential_20/lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???c
!sequential_20/lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_20/lstm_35/while/addAddV2'sequential_20_lstm_35_while_placeholder*sequential_20/lstm_35/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_20/lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential_20/lstm_35/while/add_1AddV2Dsequential_20_lstm_35_while_sequential_20_lstm_35_while_loop_counter,sequential_20/lstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$sequential_20/lstm_35/while/IdentityIdentity%sequential_20/lstm_35/while/add_1:z:0!^sequential_20/lstm_35/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_35/while/Identity_1IdentityJsequential_20_lstm_35_while_sequential_20_lstm_35_while_maximum_iterations!^sequential_20/lstm_35/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_35/while/Identity_2Identity#sequential_20/lstm_35/while/add:z:0!^sequential_20/lstm_35/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_35/while/Identity_3IdentityPsequential_20/lstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_20/lstm_35/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_35/while/Identity_4Identity2sequential_20/lstm_35/while/lstm_cell_35/mul_2:z:0!^sequential_20/lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
&sequential_20/lstm_35/while/Identity_5Identity2sequential_20/lstm_35/while/lstm_cell_35/add_1:z:0!^sequential_20/lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
 sequential_20/lstm_35/while/NoOpNoOp@^sequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?^sequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpA^sequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_20_lstm_35_while_identity-sequential_20/lstm_35/while/Identity:output:0"Y
&sequential_20_lstm_35_while_identity_1/sequential_20/lstm_35/while/Identity_1:output:0"Y
&sequential_20_lstm_35_while_identity_2/sequential_20/lstm_35/while/Identity_2:output:0"Y
&sequential_20_lstm_35_while_identity_3/sequential_20/lstm_35/while/Identity_3:output:0"Y
&sequential_20_lstm_35_while_identity_4/sequential_20/lstm_35/while/Identity_4:output:0"Y
&sequential_20_lstm_35_while_identity_5/sequential_20/lstm_35/while/Identity_5:output:0"?
Hsequential_20_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resourceJsequential_20_lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"?
Isequential_20_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resourceKsequential_20_lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"?
Gsequential_20_lstm_35_while_lstm_cell_35_matmul_readvariableop_resourceIsequential_20_lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"?
Asequential_20_lstm_35_while_sequential_20_lstm_35_strided_slice_1Csequential_20_lstm_35_while_sequential_20_lstm_35_strided_slice_1_0"?
}sequential_20_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_35_tensorarrayunstack_tensorlistfromtensorsequential_20_lstm_35_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2?
?sequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?sequential_20/lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2?
>sequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp>sequential_20/lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2?
@sequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp@sequential_20/lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?U
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_402284

inputs&
lstm_cell_36_402184:	@?&
lstm_cell_36_402186:	 ?"
lstm_cell_36_402188:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_36/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
$lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_36_402184lstm_cell_36_402186lstm_cell_36_402188*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402183n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_36_402184lstm_cell_36_402186lstm_cell_36_402188*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_402197*
condR
while_cond_402196*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402184*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402186*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_36_402188*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_36/StatefulPartitionedCall$lstm_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_35_layer_call_fn_405858

inputs
states_0
states_1
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401943o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????@:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?

?
lstm_36_while_cond_404213,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3.
*lstm_36_while_less_lstm_36_strided_slice_1D
@lstm_36_while_lstm_36_while_cond_404213___redundant_placeholder0D
@lstm_36_while_lstm_36_while_cond_404213___redundant_placeholder1D
@lstm_36_while_lstm_36_while_cond_404213___redundant_placeholder2D
@lstm_36_while_lstm_36_while_cond_404213___redundant_placeholder3
lstm_36_while_identity
?
lstm_36/while/LessLesslstm_36_while_placeholder*lstm_36_while_less_lstm_36_strided_slice_1*
T0*
_output_shapes
: [
lstm_36/while/IdentityIdentitylstm_36/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_36_while_identitylstm_36/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_402687

inputs>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_402585*
condR
while_cond_402584*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_405806

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_36_layer_call_fn_406043

inputs
states_0
states_1
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402365o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:????????? q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?U
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_401862

inputs&
lstm_cell_35_401762:	?&
lstm_cell_35_401764:	@?"
lstm_cell_35_401766:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_35/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
$lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35_401762lstm_cell_35_401764lstm_cell_35_401766*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401761n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35_401762lstm_cell_35_401764lstm_cell_35_401766*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_401775*
condR
while_cond_401774*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401762*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401764*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_35_401766*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_35/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_35/StatefulPartitionedCall$lstm_cell_35/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_404468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_404468___redundant_placeholder04
0while_while_cond_404468___redundant_placeholder14
0while_while_cond_404468___redundant_placeholder24
0while_while_cond_404468___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?8
?
while_body_403047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_lstm_cell_36_layer_call_fn_406026

inputs
states_0
states_1
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402183o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:????????? q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?"
?
while_body_402424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_36_402448_0:	@?.
while_lstm_cell_36_402450_0:	 ?*
while_lstm_cell_36_402452_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_36_402448:	@?,
while_lstm_cell_36_402450:	 ?(
while_lstm_cell_36_402452:	???*while/lstm_cell_36/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
*while/lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_36_402448_0while_lstm_cell_36_402450_0while_lstm_cell_36_402452_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402365?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity3while/lstm_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:????????? ?
while/Identity_5Identity3while/lstm_cell_36/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:????????? y

while/NoOpNoOp+^while/lstm_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_36_402448while_lstm_cell_36_402448_0"8
while_lstm_cell_36_402450while_lstm_cell_36_402450_0"8
while_lstm_cell_36_402452while_lstm_cell_36_402452_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2X
*while/lstm_cell_36/StatefulPartitionedCall*while/lstm_cell_36/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_402001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_402001___redundant_placeholder04
0while_while_cond_402001___redundant_placeholder14
0while_while_cond_402001___redundant_placeholder24
0while_while_cond_402001___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_405775

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
"__inference__traced_restore_406395
file_prefix2
 assignvariableop_dense_16_kernel: .
 assignvariableop_1_dense_16_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_35_lstm_cell_35_kernel:	?K
8assignvariableop_8_lstm_35_lstm_cell_35_recurrent_kernel:	@?;
,assignvariableop_9_lstm_35_lstm_cell_35_bias:	?B
/assignvariableop_10_lstm_36_lstm_cell_36_kernel:	@?L
9assignvariableop_11_lstm_36_lstm_cell_36_recurrent_kernel:	 ?<
-assignvariableop_12_lstm_36_lstm_cell_36_bias:	?#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_16_kernel_m: 6
(assignvariableop_16_adam_dense_16_bias_m:I
6assignvariableop_17_adam_lstm_35_lstm_cell_35_kernel_m:	?S
@assignvariableop_18_adam_lstm_35_lstm_cell_35_recurrent_kernel_m:	@?C
4assignvariableop_19_adam_lstm_35_lstm_cell_35_bias_m:	?I
6assignvariableop_20_adam_lstm_36_lstm_cell_36_kernel_m:	@?S
@assignvariableop_21_adam_lstm_36_lstm_cell_36_recurrent_kernel_m:	 ?C
4assignvariableop_22_adam_lstm_36_lstm_cell_36_bias_m:	?<
*assignvariableop_23_adam_dense_16_kernel_v: 6
(assignvariableop_24_adam_dense_16_bias_v:I
6assignvariableop_25_adam_lstm_35_lstm_cell_35_kernel_v:	?S
@assignvariableop_26_adam_lstm_35_lstm_cell_35_recurrent_kernel_v:	@?C
4assignvariableop_27_adam_lstm_35_lstm_cell_35_bias_v:	?I
6assignvariableop_28_adam_lstm_36_lstm_cell_36_kernel_v:	@?S
@assignvariableop_29_adam_lstm_36_lstm_cell_36_recurrent_kernel_v:	 ?C
4assignvariableop_30_adam_lstm_36_lstm_cell_36_bias_v:	?
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_35_lstm_cell_35_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_35_lstm_cell_35_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_35_lstm_cell_35_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_36_lstm_cell_36_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_36_lstm_cell_36_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_36_lstm_cell_36_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_16_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_16_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_35_lstm_cell_35_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_35_lstm_cell_35_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_35_lstm_cell_35_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_36_lstm_cell_36_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_36_lstm_cell_36_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_36_lstm_cell_36_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_16_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_16_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_35_lstm_cell_35_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_35_lstm_cell_35_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_35_lstm_cell_35_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_36_lstm_cell_36_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_36_lstm_cell_36_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_36_lstm_cell_36_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?8
?
while_body_402753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?A
?

lstm_36_while_body_404214,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3+
'lstm_36_while_lstm_36_strided_slice_1_0g
clstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0:	@?P
=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?K
<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
lstm_36_while_identity
lstm_36_while_identity_1
lstm_36_while_identity_2
lstm_36_while_identity_3
lstm_36_while_identity_4
lstm_36_while_identity_5)
%lstm_36_while_lstm_36_strided_slice_1e
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorL
9lstm_36_while_lstm_cell_36_matmul_readvariableop_resource:	@?N
;lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?I
:lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource:	???1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp?0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp?2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp?
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
1lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0lstm_36_while_placeholderHlstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
!lstm_36/while/lstm_cell_36/MatMulMatMul8lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
#lstm_36/while/lstm_cell_36/MatMul_1MatMullstm_36_while_placeholder_2:lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_36/while/lstm_cell_36/addAddV2+lstm_36/while/lstm_cell_36/MatMul:product:0-lstm_36/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
"lstm_36/while/lstm_cell_36/BiasAddBiasAdd"lstm_36/while/lstm_cell_36/add:z:09lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????l
*lstm_36/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_36/while/lstm_cell_36/splitSplit3lstm_36/while/lstm_cell_36/split/split_dim:output:0+lstm_36/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split?
"lstm_36/while/lstm_cell_36/SigmoidSigmoid)lstm_36/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
$lstm_36/while/lstm_cell_36/Sigmoid_1Sigmoid)lstm_36/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
lstm_36/while/lstm_cell_36/mulMul(lstm_36/while/lstm_cell_36/Sigmoid_1:y:0lstm_36_while_placeholder_3*
T0*'
_output_shapes
:????????? ?
lstm_36/while/lstm_cell_36/ReluRelu)lstm_36/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/mul_1Mul&lstm_36/while/lstm_cell_36/Sigmoid:y:0-lstm_36/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/add_1AddV2"lstm_36/while/lstm_cell_36/mul:z:0$lstm_36/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
$lstm_36/while/lstm_cell_36/Sigmoid_2Sigmoid)lstm_36/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? ?
!lstm_36/while/lstm_cell_36/Relu_1Relu$lstm_36/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
 lstm_36/while/lstm_cell_36/mul_2Mul(lstm_36/while/lstm_cell_36/Sigmoid_2:y:0/lstm_36/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
2lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_36_while_placeholder_1lstm_36_while_placeholder$lstm_36/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???U
lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_36/while/addAddV2lstm_36_while_placeholderlstm_36/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_36/while/add_1AddV2(lstm_36_while_lstm_36_while_loop_counterlstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_36/while/IdentityIdentitylstm_36/while/add_1:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_1Identity.lstm_36_while_lstm_36_while_maximum_iterations^lstm_36/while/NoOp*
T0*
_output_shapes
: q
lstm_36/while/Identity_2Identitylstm_36/while/add:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_3IdentityBlstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_36/while/NoOp*
T0*
_output_shapes
: ?
lstm_36/while/Identity_4Identity$lstm_36/while/lstm_cell_36/mul_2:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
lstm_36/while/Identity_5Identity$lstm_36/while/lstm_cell_36/add_1:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
lstm_36/while/NoOpNoOp2^lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp1^lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp3^lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_36_while_identitylstm_36/while/Identity:output:0"=
lstm_36_while_identity_1!lstm_36/while/Identity_1:output:0"=
lstm_36_while_identity_2!lstm_36/while/Identity_2:output:0"=
lstm_36_while_identity_3!lstm_36/while/Identity_3:output:0"=
lstm_36_while_identity_4!lstm_36/while/Identity_4:output:0"=
lstm_36_while_identity_5!lstm_36/while/Identity_5:output:0"P
%lstm_36_while_lstm_36_strided_slice_1'lstm_36_while_lstm_36_strided_slice_1_0"z
:lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource<lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0"|
;lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource=lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0"x
9lstm_36_while_lstm_cell_36_matmul_readvariableop_resource;lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0"?
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2f
1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp1lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp2d
0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp0lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp2h
2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp2lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_405335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_405335___redundant_placeholder04
0while_while_cond_405335___redundant_placeholder14
0while_while_cond_405335___redundant_placeholder24
0while_while_cond_405335___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
(__inference_lstm_35_layer_call_fn_404399

inputs
unknown:	?
	unknown_0:	@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_402687s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_20_layer_call_fn_402942
lstm_35_input
unknown:	?
	unknown_0:	@?
	unknown_1:	?
	unknown_2:	@?
	unknown_3:	 ?
	unknown_4:	?
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_35_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_402923o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namelstm_35_input
?
?
while_cond_402423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_402423___redundant_placeholder04
0while_while_cond_402423___redundant_placeholder14
0while_while_cond_402423___redundant_placeholder24
0while_while_cond_402423___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
)__inference_dense_16_layer_call_fn_405796

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_402880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?A
?

lstm_35_while_body_403750,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3+
'lstm_35_while_lstm_35_strided_slice_1_0g
clstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	?P
=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?K
<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
lstm_35_while_identity
lstm_35_while_identity_1
lstm_35_while_identity_2
lstm_35_while_identity_3
lstm_35_while_identity_4
lstm_35_while_identity_5)
%lstm_35_while_lstm_35_strided_slice_1e
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorL
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	?N
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	@?I
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:	???1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp?2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp?
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0lstm_35_while_placeholderHlstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
!lstm_35/while/lstm_cell_35/MatMulMatMul8lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
#lstm_35/while/lstm_cell_35/MatMul_1MatMullstm_35_while_placeholder_2:lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_35/while/lstm_cell_35/addAddV2+lstm_35/while/lstm_cell_35/MatMul:product:0-lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
"lstm_35/while/lstm_cell_35/BiasAddBiasAdd"lstm_35/while/lstm_cell_35/add:z:09lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????l
*lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_35/while/lstm_cell_35/splitSplit3lstm_35/while/lstm_cell_35/split/split_dim:output:0+lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split?
"lstm_35/while/lstm_cell_35/SigmoidSigmoid)lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
$lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid)lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
lstm_35/while/lstm_cell_35/mulMul(lstm_35/while/lstm_cell_35/Sigmoid_1:y:0lstm_35_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
lstm_35/while/lstm_cell_35/ReluRelu)lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/mul_1Mul&lstm_35/while/lstm_cell_35/Sigmoid:y:0-lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/add_1AddV2"lstm_35/while/lstm_cell_35/mul:z:0$lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
$lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid)lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@?
!lstm_35/while/lstm_cell_35/Relu_1Relu$lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/mul_2Mul(lstm_35/while/lstm_cell_35/Sigmoid_2:y:0/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
2lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_35_while_placeholder_1lstm_35_while_placeholder$lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???U
lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_35/while/addAddV2lstm_35_while_placeholderlstm_35/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_35/while/add_1AddV2(lstm_35_while_lstm_35_while_loop_counterlstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_35/while/IdentityIdentitylstm_35/while/add_1:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_1Identity.lstm_35_while_lstm_35_while_maximum_iterations^lstm_35/while/NoOp*
T0*
_output_shapes
: q
lstm_35/while/Identity_2Identitylstm_35/while/add:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_3IdentityBlstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_4Identity$lstm_35/while/lstm_cell_35/mul_2:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
lstm_35/while/Identity_5Identity$lstm_35/while/lstm_cell_35/add_1:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
lstm_35/while/NoOpNoOp2^lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1^lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp3^lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_35_while_identitylstm_35/while/Identity:output:0"=
lstm_35_while_identity_1!lstm_35/while/Identity_1:output:0"=
lstm_35_while_identity_2!lstm_35/while/Identity_2:output:0"=
lstm_35_while_identity_3!lstm_35/while/Identity_3:output:0"=
lstm_35_while_identity_4!lstm_35/while/Identity_4:output:0"=
lstm_35_while_identity_5!lstm_35/while/Identity_5:output:0"P
%lstm_35_while_lstm_35_strided_slice_1'lstm_35_while_lstm_35_strided_slice_1_0"z
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"|
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"x
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"?
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2f
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2d
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2h
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?R
?
'sequential_20_lstm_36_while_body_401585H
Dsequential_20_lstm_36_while_sequential_20_lstm_36_while_loop_counterN
Jsequential_20_lstm_36_while_sequential_20_lstm_36_while_maximum_iterations+
'sequential_20_lstm_36_while_placeholder-
)sequential_20_lstm_36_while_placeholder_1-
)sequential_20_lstm_36_while_placeholder_2-
)sequential_20_lstm_36_while_placeholder_3G
Csequential_20_lstm_36_while_sequential_20_lstm_36_strided_slice_1_0?
sequential_20_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_36_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_20_lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0:	@?^
Ksequential_20_lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?Y
Jsequential_20_lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0:	?(
$sequential_20_lstm_36_while_identity*
&sequential_20_lstm_36_while_identity_1*
&sequential_20_lstm_36_while_identity_2*
&sequential_20_lstm_36_while_identity_3*
&sequential_20_lstm_36_while_identity_4*
&sequential_20_lstm_36_while_identity_5E
Asequential_20_lstm_36_while_sequential_20_lstm_36_strided_slice_1?
}sequential_20_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_36_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_20_lstm_36_while_lstm_cell_36_matmul_readvariableop_resource:	@?\
Isequential_20_lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?W
Hsequential_20_lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource:	????sequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp?>sequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp?@sequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp?
Msequential_20/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
?sequential_20/lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_20_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_36_tensorarrayunstack_tensorlistfromtensor_0'sequential_20_lstm_36_while_placeholderVsequential_20/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
>sequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOpIsequential_20_lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
/sequential_20/lstm_36/while/lstm_cell_36/MatMulMatMulFsequential_20/lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
@sequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOpKsequential_20_lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
1sequential_20/lstm_36/while/lstm_cell_36/MatMul_1MatMul)sequential_20_lstm_36_while_placeholder_2Hsequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_20/lstm_36/while/lstm_cell_36/addAddV29sequential_20/lstm_36/while/lstm_cell_36/MatMul:product:0;sequential_20/lstm_36/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
?sequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOpJsequential_20_lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
0sequential_20/lstm_36/while/lstm_cell_36/BiasAddBiasAdd0sequential_20/lstm_36/while/lstm_cell_36/add:z:0Gsequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
8sequential_20/lstm_36/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
.sequential_20/lstm_36/while/lstm_cell_36/splitSplitAsequential_20/lstm_36/while/lstm_cell_36/split/split_dim:output:09sequential_20/lstm_36/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split?
0sequential_20/lstm_36/while/lstm_cell_36/SigmoidSigmoid7sequential_20/lstm_36/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
2sequential_20/lstm_36/while/lstm_cell_36/Sigmoid_1Sigmoid7sequential_20/lstm_36/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
,sequential_20/lstm_36/while/lstm_cell_36/mulMul6sequential_20/lstm_36/while/lstm_cell_36/Sigmoid_1:y:0)sequential_20_lstm_36_while_placeholder_3*
T0*'
_output_shapes
:????????? ?
-sequential_20/lstm_36/while/lstm_cell_36/ReluRelu7sequential_20/lstm_36/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
.sequential_20/lstm_36/while/lstm_cell_36/mul_1Mul4sequential_20/lstm_36/while/lstm_cell_36/Sigmoid:y:0;sequential_20/lstm_36/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
.sequential_20/lstm_36/while/lstm_cell_36/add_1AddV20sequential_20/lstm_36/while/lstm_cell_36/mul:z:02sequential_20/lstm_36/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
2sequential_20/lstm_36/while/lstm_cell_36/Sigmoid_2Sigmoid7sequential_20/lstm_36/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? ?
/sequential_20/lstm_36/while/lstm_cell_36/Relu_1Relu2sequential_20/lstm_36/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
.sequential_20/lstm_36/while/lstm_cell_36/mul_2Mul6sequential_20/lstm_36/while/lstm_cell_36/Sigmoid_2:y:0=sequential_20/lstm_36/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
@sequential_20/lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_20_lstm_36_while_placeholder_1'sequential_20_lstm_36_while_placeholder2sequential_20/lstm_36/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???c
!sequential_20/lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_20/lstm_36/while/addAddV2'sequential_20_lstm_36_while_placeholder*sequential_20/lstm_36/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_20/lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential_20/lstm_36/while/add_1AddV2Dsequential_20_lstm_36_while_sequential_20_lstm_36_while_loop_counter,sequential_20/lstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: ?
$sequential_20/lstm_36/while/IdentityIdentity%sequential_20/lstm_36/while/add_1:z:0!^sequential_20/lstm_36/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_36/while/Identity_1IdentityJsequential_20_lstm_36_while_sequential_20_lstm_36_while_maximum_iterations!^sequential_20/lstm_36/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_36/while/Identity_2Identity#sequential_20/lstm_36/while/add:z:0!^sequential_20/lstm_36/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_36/while/Identity_3IdentityPsequential_20/lstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_20/lstm_36/while/NoOp*
T0*
_output_shapes
: ?
&sequential_20/lstm_36/while/Identity_4Identity2sequential_20/lstm_36/while/lstm_cell_36/mul_2:z:0!^sequential_20/lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
&sequential_20/lstm_36/while/Identity_5Identity2sequential_20/lstm_36/while/lstm_cell_36/add_1:z:0!^sequential_20/lstm_36/while/NoOp*
T0*'
_output_shapes
:????????? ?
 sequential_20/lstm_36/while/NoOpNoOp@^sequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp?^sequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOpA^sequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_20_lstm_36_while_identity-sequential_20/lstm_36/while/Identity:output:0"Y
&sequential_20_lstm_36_while_identity_1/sequential_20/lstm_36/while/Identity_1:output:0"Y
&sequential_20_lstm_36_while_identity_2/sequential_20/lstm_36/while/Identity_2:output:0"Y
&sequential_20_lstm_36_while_identity_3/sequential_20/lstm_36/while/Identity_3:output:0"Y
&sequential_20_lstm_36_while_identity_4/sequential_20/lstm_36/while/Identity_4:output:0"Y
&sequential_20_lstm_36_while_identity_5/sequential_20/lstm_36/while/Identity_5:output:0"?
Hsequential_20_lstm_36_while_lstm_cell_36_biasadd_readvariableop_resourceJsequential_20_lstm_36_while_lstm_cell_36_biasadd_readvariableop_resource_0"?
Isequential_20_lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resourceKsequential_20_lstm_36_while_lstm_cell_36_matmul_1_readvariableop_resource_0"?
Gsequential_20_lstm_36_while_lstm_cell_36_matmul_readvariableop_resourceIsequential_20_lstm_36_while_lstm_cell_36_matmul_readvariableop_resource_0"?
Asequential_20_lstm_36_while_sequential_20_lstm_36_strided_slice_1Csequential_20_lstm_36_while_sequential_20_lstm_36_strided_slice_1_0"?
}sequential_20_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_36_tensorarrayunstack_tensorlistfromtensorsequential_20_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2?
?sequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp?sequential_20/lstm_36/while/lstm_cell_36/BiasAdd/ReadVariableOp2?
>sequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp>sequential_20/lstm_36/while/lstm_cell_36/MatMul/ReadVariableOp2?
@sequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp@sequential_20/lstm_36/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
d
+__inference_dropout_16_layer_call_fn_405770

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402972o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?"
?
while_body_401775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_35_401799_0:	?.
while_lstm_cell_35_401801_0:	@?*
while_lstm_cell_35_401803_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_35_401799:	?,
while_lstm_cell_35_401801:	@?(
while_lstm_cell_35_401803:	???*while/lstm_cell_35/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
*while/lstm_cell_35/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35_401799_0while_lstm_cell_35_401801_0while_lstm_cell_35_401803_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_401761?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_35/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity3while/lstm_cell_35/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????@?
while/Identity_5Identity3while/lstm_cell_35/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????@y

while/NoOpNoOp+^while/lstm_cell_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_35_401799while_lstm_cell_35_401799_0"8
while_lstm_cell_35_401801while_lstm_cell_35_401801_0"8
while_lstm_cell_35_401803while_lstm_cell_35_401803_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2X
*while/lstm_cell_35/StatefulPartitionedCall*while/lstm_cell_35/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_lstm_36_layer_call_fn_405094
inputs_0
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_402511o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
while_cond_402584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_402584___redundant_placeholder04
0while_while_cond_402584___redundant_placeholder14
0while_while_cond_402584___redundant_placeholder24
0while_while_cond_402584___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?7
?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_402365

inputs

states
states_11
matmul_readvariableop_resource:	@?3
 matmul_1_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? N
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:????????? :????????? : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?
?
while_cond_405657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_405657___redundant_placeholder04
0while_while_cond_405657___redundant_placeholder14
0while_while_cond_405657___redundant_placeholder24
0while_while_cond_405657___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
__inference_loss_fn_5_406176S
Dlstm_36_lstm_cell_36_bias_regularizer_square_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpDlstm_36_lstm_cell_36_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentity-lstm_36/lstm_cell_36/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp
??
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_404016

inputsF
3lstm_35_lstm_cell_35_matmul_readvariableop_resource:	?H
5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource:	@?C
4lstm_35_lstm_cell_35_biasadd_readvariableop_resource:	?F
3lstm_36_lstm_cell_36_matmul_readvariableop_resource:	@?H
5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource:	 ?C
4lstm_36_lstm_cell_36_biasadd_readvariableop_resource:	?9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp?*lstm_35/lstm_cell_35/MatMul/ReadVariableOp?,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_35/while?+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp?*lstm_36/lstm_cell_36/MatMul/ReadVariableOp?,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/whileC
lstm_35/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_sliceStridedSlicelstm_35/Shape:output:0$lstm_35/strided_slice/stack:output:0&lstm_35/strided_slice/stack_1:output:0&lstm_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_35/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
lstm_35/zeros/packedPacklstm_35/strided_slice:output:0lstm_35/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_35/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_35/zerosFilllstm_35/zeros/packed:output:0lstm_35/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@Z
lstm_35/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@?
lstm_35/zeros_1/packedPacklstm_35/strided_slice:output:0!lstm_35/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_35/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_35/zeros_1Filllstm_35/zeros_1/packed:output:0lstm_35/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@k
lstm_35/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_35/transpose	Transposeinputslstm_35/transpose/perm:output:0*
T0*+
_output_shapes
:?????????T
lstm_35/Shape_1Shapelstm_35/transpose:y:0*
T0*
_output_shapes
:g
lstm_35/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_35/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_1StridedSlicelstm_35/Shape_1:output:0&lstm_35/strided_slice_1/stack:output:0(lstm_35/strided_slice_1/stack_1:output:0(lstm_35/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_35/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_35/TensorArrayV2TensorListReserve,lstm_35/TensorArrayV2/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
=lstm_35/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/lstm_35/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_35/transpose:y:0Flstm_35/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???g
lstm_35/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_35/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_2StridedSlicelstm_35/transpose:y:0&lstm_35/strided_slice_2/stack:output:0(lstm_35/strided_slice_2/stack_1:output:0(lstm_35/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
*lstm_35/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_35/lstm_cell_35/MatMulMatMul lstm_35/strided_slice_2:output:02lstm_35/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_35/lstm_cell_35/MatMul_1MatMullstm_35/zeros:output:04lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_35/lstm_cell_35/addAddV2%lstm_35/lstm_cell_35/MatMul:product:0'lstm_35/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_35/lstm_cell_35/BiasAddBiasAddlstm_35/lstm_cell_35/add:z:03lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
$lstm_35/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_35/lstm_cell_35/splitSplit-lstm_35/lstm_cell_35/split/split_dim:output:0%lstm_35/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split~
lstm_35/lstm_cell_35/SigmoidSigmoid#lstm_35/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/Sigmoid_1Sigmoid#lstm_35/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mulMul"lstm_35/lstm_cell_35/Sigmoid_1:y:0lstm_35/zeros_1:output:0*
T0*'
_output_shapes
:?????????@x
lstm_35/lstm_cell_35/ReluRelu#lstm_35/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mul_1Mul lstm_35/lstm_cell_35/Sigmoid:y:0'lstm_35/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/add_1AddV2lstm_35/lstm_cell_35/mul:z:0lstm_35/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/Sigmoid_2Sigmoid#lstm_35/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@u
lstm_35/lstm_cell_35/Relu_1Relulstm_35/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_35/lstm_cell_35/mul_2Mul"lstm_35/lstm_cell_35/Sigmoid_2:y:0)lstm_35/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@v
%lstm_35/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
lstm_35/TensorArrayV2_1TensorListReserve.lstm_35/TensorArrayV2_1/element_shape:output:0 lstm_35/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???N
lstm_35/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_35/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????\
lstm_35/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_35/whileWhile#lstm_35/while/loop_counter:output:0)lstm_35/while/maximum_iterations:output:0lstm_35/time:output:0 lstm_35/TensorArrayV2_1:handle:0lstm_35/zeros:output:0lstm_35/zeros_1:output:0 lstm_35/strided_slice_1:output:0?lstm_35/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_35_lstm_cell_35_matmul_readvariableop_resource5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_35_while_body_403750*%
condR
lstm_35_while_cond_403749*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
8lstm_35/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
*lstm_35/TensorArrayV2Stack/TensorListStackTensorListStacklstm_35/while:output:3Alstm_35/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????@*
element_dtype0p
lstm_35/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????i
lstm_35/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_35/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_35/strided_slice_3StridedSlice3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_35/strided_slice_3/stack:output:0(lstm_35/strided_slice_3/stack_1:output:0(lstm_35/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskm
lstm_35/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_35/transpose_1	Transpose3lstm_35/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_35/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????@c
lstm_35/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    T
lstm_36/ShapeShapelstm_35/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_sliceStridedSlicelstm_36/Shape:output:0$lstm_36/strided_slice/stack:output:0&lstm_36/strided_slice/stack_1:output:0&lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/zeros/packedPacklstm_36/strided_slice:output:0lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_36/zerosFilllstm_36/zeros/packed:output:0lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:????????? Z
lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/zeros_1/packedPacklstm_36/strided_slice:output:0!lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_36/zeros_1Filllstm_36/zeros_1/packed:output:0lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? k
lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_36/transpose	Transposelstm_35/transpose_1:y:0lstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:?????????@T
lstm_36/Shape_1Shapelstm_36/transpose:y:0*
T0*
_output_shapes
:g
lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_1StridedSlicelstm_36/Shape_1:output:0&lstm_36/strided_slice_1/stack:output:0(lstm_36/strided_slice_1/stack_1:output:0(lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_36/TensorArrayV2TensorListReserve,lstm_36/TensorArrayV2/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_36/transpose:y:0Flstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???g
lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_2StridedSlicelstm_36/transpose:y:0&lstm_36/strided_slice_2/stack:output:0(lstm_36/strided_slice_2/stack_1:output:0(lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
*lstm_36/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_36/lstm_cell_36/MatMulMatMul lstm_36/strided_slice_2:output:02lstm_36/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_36/lstm_cell_36/MatMul_1MatMullstm_36/zeros:output:04lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_36/lstm_cell_36/addAddV2%lstm_36/lstm_cell_36/MatMul:product:0'lstm_36/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_36/lstm_cell_36/BiasAddBiasAddlstm_36/lstm_cell_36/add:z:03lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
$lstm_36/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_36/lstm_cell_36/splitSplit-lstm_36/lstm_cell_36/split/split_dim:output:0%lstm_36/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split~
lstm_36/lstm_cell_36/SigmoidSigmoid#lstm_36/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/Sigmoid_1Sigmoid#lstm_36/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mulMul"lstm_36/lstm_cell_36/Sigmoid_1:y:0lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:????????? x
lstm_36/lstm_cell_36/ReluRelu#lstm_36/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mul_1Mul lstm_36/lstm_cell_36/Sigmoid:y:0'lstm_36/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/add_1AddV2lstm_36/lstm_cell_36/mul:z:0lstm_36/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/Sigmoid_2Sigmoid#lstm_36/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? u
lstm_36/lstm_cell_36/Relu_1Relulstm_36/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_36/lstm_cell_36/mul_2Mul"lstm_36/lstm_cell_36/Sigmoid_2:y:0)lstm_36/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? v
%lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
lstm_36/TensorArrayV2_1TensorListReserve.lstm_36/TensorArrayV2_1/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???N
lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????\
lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_36/whileWhile#lstm_36/while/loop_counter:output:0)lstm_36/while/maximum_iterations:output:0lstm_36/time:output:0 lstm_36/TensorArrayV2_1:handle:0lstm_36/zeros:output:0lstm_36/zeros_1:output:0 lstm_36/strided_slice_1:output:0?lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_36_lstm_cell_36_matmul_readvariableop_resource5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_36_while_body_403889*%
condR
lstm_36_while_cond_403888*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
*lstm_36/TensorArrayV2Stack/TensorListStackTensorListStacklstm_36/while:output:3Alstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0p
lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????i
lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_36/strided_slice_3StridedSlice3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_36/strided_slice_3/stack:output:0(lstm_36/strided_slice_3/stack_1:output:0(lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskm
lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_36/transpose_1	Transpose3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? c
lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_16/IdentityIdentity lstm_36/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? ?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_35_lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp5lstm_35_lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_35_lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_16/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp,^lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+^lstm_35/lstm_cell_35/MatMul/ReadVariableOp-^lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm_35/while,^lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp+^lstm_36/lstm_cell_36/MatMul/ReadVariableOp-^lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm_36/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2Z
+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp+lstm_35/lstm_cell_35/BiasAdd/ReadVariableOp2X
*lstm_35/lstm_cell_35/MatMul/ReadVariableOp*lstm_35/lstm_cell_35/MatMul/ReadVariableOp2\
,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp,lstm_35/lstm_cell_35/MatMul_1/ReadVariableOp2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2
lstm_35/whilelstm_35/while2Z
+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp+lstm_36/lstm_cell_36/BiasAdd/ReadVariableOp2X
*lstm_36/lstm_cell_36/MatMul/ReadVariableOp*lstm_36/lstm_cell_36/MatMul/ReadVariableOp2\
,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp,lstm_36/lstm_cell_36/MatMul_1/ReadVariableOp2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2
lstm_36/whilelstm_36/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?8
?
while_body_404791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_35_matmul_readvariableop_resource_0:	?H
5while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?C
4while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_35_matmul_readvariableop_resource:	?F
3while_lstm_cell_35_matmul_1_readvariableop_resource:	@?A
2while_lstm_cell_35_biasadd_readvariableop_resource:	???)while/lstm_cell_35/BiasAdd/ReadVariableOp?(while/lstm_cell_35/MatMul/ReadVariableOp?*while/lstm_cell_35/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_35/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_35/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_35/addAddV2#while/lstm_cell_35/MatMul:product:0%while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_35/BiasAddBiasAddwhile/lstm_cell_35/add:z:01while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_35/splitSplit+while/lstm_cell_35/split/split_dim:output:0#while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitz
while/lstm_cell_35/SigmoidSigmoid!while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_1Sigmoid!while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mulMul while/lstm_cell_35/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@t
while/lstm_cell_35/ReluRelu!while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_1Mulwhile/lstm_cell_35/Sigmoid:y:0%while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/add_1AddV2while/lstm_cell_35/mul:z:0while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@|
while/lstm_cell_35/Sigmoid_2Sigmoid!while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@q
while/lstm_cell_35/Relu_1Reluwhile/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_35/mul_2Mul while/lstm_cell_35/Sigmoid_2:y:0'while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_35/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@y
while/Identity_5Identitywhile/lstm_cell_35/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????@?

while/NoOpNoOp*^while/lstm_cell_35/BiasAdd/ReadVariableOp)^while/lstm_cell_35/MatMul/ReadVariableOp+^while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_35_biasadd_readvariableop_resource4while_lstm_cell_35_biasadd_readvariableop_resource_0"l
3while_lstm_cell_35_matmul_1_readvariableop_resource5while_lstm_cell_35_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_35_matmul_readvariableop_resource3while_lstm_cell_35_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2V
)while/lstm_cell_35/BiasAdd/ReadVariableOp)while/lstm_cell_35/BiasAdd/ReadVariableOp2T
(while/lstm_cell_35/MatMul/ReadVariableOp(while/lstm_cell_35/MatMul/ReadVariableOp2X
*while/lstm_cell_35/MatMul_1/ReadVariableOp*while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?h
?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404732
inputs_0>
+lstm_cell_35_matmul_readvariableop_resource:	?@
-lstm_cell_35_matmul_1_readvariableop_resource:	@?;
,lstm_cell_35_biasadd_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_35/BiasAdd/ReadVariableOp?"lstm_cell_35/MatMul/ReadVariableOp?$lstm_cell_35/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"lstm_cell_35/MatMul/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_35/MatMulMatMulstrided_slice_2:output:0*lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_35/MatMul_1MatMulzeros:output:0,lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_35/addAddV2lstm_cell_35/MatMul:product:0lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_35/BiasAddBiasAddlstm_cell_35/add:z:0+lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_35/splitSplit%lstm_cell_35/split/split_dim:output:0lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_splitn
lstm_cell_35/SigmoidSigmoidlstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_1Sigmoidlstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@w
lstm_cell_35/mulMullstm_cell_35/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@h
lstm_cell_35/ReluRelulstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_1Mullstm_cell_35/Sigmoid:y:0lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@{
lstm_cell_35/add_1AddV2lstm_cell_35/mul:z:0lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@p
lstm_cell_35/Sigmoid_2Sigmoidlstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@e
lstm_cell_35/Relu_1Relulstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_35/mul_2Mullstm_cell_35/Sigmoid_2:y:0!lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_35_matmul_readvariableop_resource-lstm_cell_35_matmul_1_readvariableop_resource,lstm_cell_35_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_404630*
condR
while_cond_404629*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_35_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_35_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_35/BiasAdd/ReadVariableOp#^lstm_cell_35/MatMul/ReadVariableOp%^lstm_cell_35/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_35/BiasAdd/ReadVariableOp#lstm_cell_35/BiasAdd/ReadVariableOp2H
"lstm_cell_35/MatMul/ReadVariableOp"lstm_cell_35/MatMul/ReadVariableOp2L
$lstm_cell_35/MatMul_1/ReadVariableOp$lstm_cell_35/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
(__inference_lstm_36_layer_call_fn_405105

inputs
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_402855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_403149

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_403047*
condR
while_cond_403046*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_405969Y
Flstm_35_lstm_cell_35_kernel_regularizer_square_readvariableop_resource:	?
identity??=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFlstm_35_lstm_cell_35_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/lstm_35/lstm_cell_35/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_402880

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
while_body_405175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_405496
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_405496___redundant_placeholder04
0while_while_cond_405496___redundant_placeholder14
0while_while_cond_405496___redundant_placeholder24
0while_while_cond_405496___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?	
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_402972

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?Q
?
I__inference_sequential_20_layer_call_and_return_conditional_losses_403424

inputs!
lstm_35_403367:	?!
lstm_35_403369:	@?
lstm_35_403371:	?!
lstm_36_403374:	@?!
lstm_36_403376:	 ?
lstm_36_403378:	?!
dense_16_403382: 
dense_16_403384:
identity?? dense_16/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?lstm_35/StatefulPartitionedCall?;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp?Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp?lstm_36/StatefulPartitionedCall?;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?
lstm_35/StatefulPartitionedCallStatefulPartitionedCallinputslstm_35_403367lstm_35_403369lstm_35_403371*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_35_layer_call_and_return_conditional_losses_403332?
lstm_36/StatefulPartitionedCallStatefulPartitionedCall(lstm_35/StatefulPartitionedCall:output:0lstm_36_403374lstm_36_403376lstm_36_403378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_403149?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402972?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_403382dense_16_403384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_402880?
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403367*
_output_shapes
:	?*
dtype0?
.lstm_35/lstm_cell_35/kernel/Regularizer/SquareSquareElstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?~
-lstm_35/lstm_cell_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_35/lstm_cell_35/kernel/Regularizer/SumSum2lstm_35/lstm_cell_35/kernel/Regularizer/Square:y:06lstm_35/lstm_cell_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_35/lstm_cell_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_35/lstm_cell_35/kernel/Regularizer/mulMul6lstm_35/lstm_cell_35/kernel/Regularizer/mul/x:output:04lstm_35/lstm_cell_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403369*
_output_shapes
:	@?*
dtype0?
8lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SquareSquareOlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@??
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/SumSum<lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square:y:0@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mulMul@lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/mul/x:output:0>lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_35_403371*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403374*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403376*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_36_403378*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall ^lstm_35/StatefulPartitionedCall<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp>^lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOpH^lstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp ^lstm_36/StatefulPartitionedCall<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2B
lstm_35/StatefulPartitionedCalllstm_35/StatefulPartitionedCall2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp2~
=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp=lstm_35/lstm_cell_35/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_35/lstm_cell_35/recurrent_kernel/Regularizer/Square/ReadVariableOp2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
lstm_35_while_cond_404074,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3.
*lstm_35_while_less_lstm_35_strided_slice_1D
@lstm_35_while_lstm_35_while_cond_404074___redundant_placeholder0D
@lstm_35_while_lstm_35_while_cond_404074___redundant_placeholder1D
@lstm_35_while_lstm_35_while_cond_404074___redundant_placeholder2D
@lstm_35_while_lstm_35_while_cond_404074___redundant_placeholder3
lstm_35_while_identity
?
lstm_35/while/LessLesslstm_35_while_placeholder*lstm_35_while_less_lstm_35_strided_slice_1*
T0*
_output_shapes
: [
lstm_35/while/IdentityIdentitylstm_35/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_35_while_identitylstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
(__inference_lstm_36_layer_call_fn_405083
inputs_0
unknown:	@?
	unknown_0:	 ?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_36_layer_call_and_return_conditional_losses_402284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?A
?

lstm_35_while_body_404075,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3+
'lstm_35_while_lstm_35_strided_slice_1_0g
clstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0:	?P
=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0:	@?K
<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0:	?
lstm_35_while_identity
lstm_35_while_identity_1
lstm_35_while_identity_2
lstm_35_while_identity_3
lstm_35_while_identity_4
lstm_35_while_identity_5)
%lstm_35_while_lstm_35_strided_slice_1e
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorL
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource:	?N
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource:	@?I
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource:	???1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp?0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp?2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp?
?lstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1lstm_35/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0lstm_35_while_placeholderHlstm_35/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOpReadVariableOp;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
!lstm_35/while/lstm_cell_35/MatMulMatMul8lstm_35/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOpReadVariableOp=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
#lstm_35/while/lstm_cell_35/MatMul_1MatMullstm_35_while_placeholder_2:lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_35/while/lstm_cell_35/addAddV2+lstm_35/while/lstm_cell_35/MatMul:product:0-lstm_35/while/lstm_cell_35/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOpReadVariableOp<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
"lstm_35/while/lstm_cell_35/BiasAddBiasAdd"lstm_35/while/lstm_cell_35/add:z:09lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????l
*lstm_35/while/lstm_cell_35/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_35/while/lstm_cell_35/splitSplit3lstm_35/while/lstm_cell_35/split/split_dim:output:0+lstm_35/while/lstm_cell_35/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split?
"lstm_35/while/lstm_cell_35/SigmoidSigmoid)lstm_35/while/lstm_cell_35/split:output:0*
T0*'
_output_shapes
:?????????@?
$lstm_35/while/lstm_cell_35/Sigmoid_1Sigmoid)lstm_35/while/lstm_cell_35/split:output:1*
T0*'
_output_shapes
:?????????@?
lstm_35/while/lstm_cell_35/mulMul(lstm_35/while/lstm_cell_35/Sigmoid_1:y:0lstm_35_while_placeholder_3*
T0*'
_output_shapes
:?????????@?
lstm_35/while/lstm_cell_35/ReluRelu)lstm_35/while/lstm_cell_35/split:output:2*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/mul_1Mul&lstm_35/while/lstm_cell_35/Sigmoid:y:0-lstm_35/while/lstm_cell_35/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/add_1AddV2"lstm_35/while/lstm_cell_35/mul:z:0$lstm_35/while/lstm_cell_35/mul_1:z:0*
T0*'
_output_shapes
:?????????@?
$lstm_35/while/lstm_cell_35/Sigmoid_2Sigmoid)lstm_35/while/lstm_cell_35/split:output:3*
T0*'
_output_shapes
:?????????@?
!lstm_35/while/lstm_cell_35/Relu_1Relu$lstm_35/while/lstm_cell_35/add_1:z:0*
T0*'
_output_shapes
:?????????@?
 lstm_35/while/lstm_cell_35/mul_2Mul(lstm_35/while/lstm_cell_35/Sigmoid_2:y:0/lstm_35/while/lstm_cell_35/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@?
2lstm_35/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_35_while_placeholder_1lstm_35_while_placeholder$lstm_35/while/lstm_cell_35/mul_2:z:0*
_output_shapes
: *
element_dtype0:???U
lstm_35/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_35/while/addAddV2lstm_35_while_placeholderlstm_35/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_35/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_35/while/add_1AddV2(lstm_35_while_lstm_35_while_loop_counterlstm_35/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_35/while/IdentityIdentitylstm_35/while/add_1:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_1Identity.lstm_35_while_lstm_35_while_maximum_iterations^lstm_35/while/NoOp*
T0*
_output_shapes
: q
lstm_35/while/Identity_2Identitylstm_35/while/add:z:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_3IdentityBlstm_35/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_35/while/NoOp*
T0*
_output_shapes
: ?
lstm_35/while/Identity_4Identity$lstm_35/while/lstm_cell_35/mul_2:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
lstm_35/while/Identity_5Identity$lstm_35/while/lstm_cell_35/add_1:z:0^lstm_35/while/NoOp*
T0*'
_output_shapes
:?????????@?
lstm_35/while/NoOpNoOp2^lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1^lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp3^lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_35_while_identitylstm_35/while/Identity:output:0"=
lstm_35_while_identity_1!lstm_35/while/Identity_1:output:0"=
lstm_35_while_identity_2!lstm_35/while/Identity_2:output:0"=
lstm_35_while_identity_3!lstm_35/while/Identity_3:output:0"=
lstm_35_while_identity_4!lstm_35/while/Identity_4:output:0"=
lstm_35_while_identity_5!lstm_35/while/Identity_5:output:0"P
%lstm_35_while_lstm_35_strided_slice_1'lstm_35_while_lstm_35_strided_slice_1_0"z
:lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource<lstm_35_while_lstm_cell_35_biasadd_readvariableop_resource_0"|
;lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource=lstm_35_while_lstm_cell_35_matmul_1_readvariableop_resource_0"x
9lstm_35_while_lstm_cell_35_matmul_readvariableop_resource;lstm_35_while_lstm_cell_35_matmul_readvariableop_resource_0"?
alstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensorclstm_35_while_tensorarrayv2read_tensorlistgetitem_lstm_35_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????@:?????????@: : : : : 2f
1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp1lstm_35/while/lstm_cell_35/BiasAdd/ReadVariableOp2d
0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp0lstm_35/while/lstm_cell_35/MatMul/ReadVariableOp2h
2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp2lstm_35/while/lstm_cell_35/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_401774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_401774___redundant_placeholder04
0while_while_cond_401774___redundant_placeholder14
0while_while_cond_401774___redundant_placeholder24
0while_while_cond_401774___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?h
?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405438
inputs_0>
+lstm_cell_36_matmul_readvariableop_resource:	@?@
-lstm_cell_36_matmul_1_readvariableop_resource:	 ?;
,lstm_cell_36_biasadd_readvariableop_resource:	?
identity??;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp?=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp?Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp?#lstm_cell_36/BiasAdd/ReadVariableOp?"lstm_cell_36/MatMul/ReadVariableOp?$lstm_cell_36/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitn
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? w
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? h
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? {
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? p
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? e
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_405336*
condR
while_cond_405335*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
.lstm_36/lstm_cell_36/kernel/Regularizer/SquareSquareElstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@?~
-lstm_36/lstm_cell_36/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+lstm_36/lstm_cell_36/kernel/Regularizer/SumSum2lstm_36/lstm_cell_36/kernel/Regularizer/Square:y:06lstm_36/lstm_cell_36/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-lstm_36/lstm_cell_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
+lstm_36/lstm_cell_36/kernel/Regularizer/mulMul6lstm_36/lstm_cell_36/kernel/Regularizer/mul/x:output:04lstm_36/lstm_cell_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
8lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SquareSquareOlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ??
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/SumSum<lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square:y:0@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
5lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mulMul@lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/mul/x:output:0>lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_36/lstm_cell_36/bias/Regularizer/SquareSquareClstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_36/lstm_cell_36/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_36/lstm_cell_36/bias/Regularizer/SumSum0lstm_36/lstm_cell_36/bias/Regularizer/Square:y:04lstm_36/lstm_cell_36/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_36/lstm_cell_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_36/lstm_cell_36/bias/Regularizer/mulMul4lstm_36/lstm_cell_36/bias/Regularizer/mul/x:output:02lstm_36/lstm_cell_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp<^lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp>^lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOpH^lstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2z
;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp;lstm_36/lstm_cell_36/bias/Regularizer/Square/ReadVariableOp2~
=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp=lstm_36/lstm_cell_36/kernel/Regularizer/Square/ReadVariableOp2?
Glstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOpGlstm_36/lstm_cell_36/recurrent_kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
while_cond_405174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_405174___redundant_placeholder04
0while_while_cond_405174___redundant_placeholder14
0while_while_cond_405174___redundant_placeholder24
0while_while_cond_405174___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?8
?
while_body_405658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?8
?
while_body_405497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	@?H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	 ?C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	@?F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	 ?A
2while_lstm_cell_36_biasadd_readvariableop_resource:	???)while/lstm_cell_36/BiasAdd/ReadVariableOp?(while/lstm_cell_36/MatMul/ReadVariableOp?*while/lstm_cell_36/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype0?
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_splitz
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? t
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:????????? |
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:????????? q
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:????????? ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:????????? y
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:????????? ?

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :????????? :????????? : : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?

?
lstm_35_while_cond_403749,
(lstm_35_while_lstm_35_while_loop_counter2
.lstm_35_while_lstm_35_while_maximum_iterations
lstm_35_while_placeholder
lstm_35_while_placeholder_1
lstm_35_while_placeholder_2
lstm_35_while_placeholder_3.
*lstm_35_while_less_lstm_35_strided_slice_1D
@lstm_35_while_lstm_35_while_cond_403749___redundant_placeholder0D
@lstm_35_while_lstm_35_while_cond_403749___redundant_placeholder1D
@lstm_35_while_lstm_35_while_cond_403749___redundant_placeholder2D
@lstm_35_while_lstm_35_while_cond_403749___redundant_placeholder3
lstm_35_while_identity
?
lstm_35/while/LessLesslstm_35_while_placeholder*lstm_35_while_less_lstm_35_strided_slice_1*
T0*
_output_shapes
: [
lstm_35/while/IdentityIdentitylstm_35/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_35_while_identitylstm_35/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
__inference_loss_fn_2_405991S
Dlstm_35_lstm_cell_35_bias_regularizer_square_readvariableop_resource:	?
identity??;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp?
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpDlstm_35_lstm_cell_35_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,lstm_35/lstm_cell_35/bias/Regularizer/SquareSquareClstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?u
+lstm_35/lstm_cell_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)lstm_35/lstm_cell_35/bias/Regularizer/SumSum0lstm_35/lstm_cell_35/bias/Regularizer/Square:y:04lstm_35/lstm_cell_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+lstm_35/lstm_cell_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
)lstm_35/lstm_cell_35/bias/Regularizer/mulMul4lstm_35/lstm_cell_35/bias/Regularizer/mul/x:output:02lstm_35/lstm_cell_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentity-lstm_35/lstm_cell_35/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp<^lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp;lstm_35/lstm_cell_35/bias/Regularizer/Square/ReadVariableOp
?
G
+__inference_dropout_16_layer_call_fn_405765

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_402868`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
.__inference_sequential_20_layer_call_fn_403670

inputs
unknown:	?
	unknown_0:	@?
	unknown_1:	?
	unknown_2:	@?
	unknown_3:	 ?
	unknown_4:	?
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_402923o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_35_input:
serving_default_lstm_35_input:0?????????<
dense_160
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
p__call__
*q&call_and_return_all_conditional_losses
r_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratem`ma&mb'mc(md)me*mf+mgvhvi&vj'vk(vl)vm*vn+vo"
	optimizer
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
p__call__
r_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
?
1
state_size

&kernel
'recurrent_kernel
(bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
6
~0
1
?2"
trackable_list_wrapper
?

6states
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
?
<
state_size

)kernel
*recurrent_kernel
+bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?

Astates
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_16/kernel
:2dense_16/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	?2lstm_35/lstm_cell_35/kernel
8:6	@?2%lstm_35/lstm_cell_35/recurrent_kernel
(:&?2lstm_35/lstm_cell_35/bias
.:,	@?2lstm_36/lstm_cell_36/kernel
8:6	 ?2%lstm_36/lstm_cell_36/recurrent_kernel
(:&?2lstm_36/lstm_cell_36/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
6
~0
1
?2"
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
2	variables
3trainable_variables
4regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
=	variables
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
6
~0
1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
&:$ 2Adam/dense_16/kernel/m
 :2Adam/dense_16/bias/m
3:1	?2"Adam/lstm_35/lstm_cell_35/kernel/m
=:;	@?2,Adam/lstm_35/lstm_cell_35/recurrent_kernel/m
-:+?2 Adam/lstm_35/lstm_cell_35/bias/m
3:1	@?2"Adam/lstm_36/lstm_cell_36/kernel/m
=:;	 ?2,Adam/lstm_36/lstm_cell_36/recurrent_kernel/m
-:+?2 Adam/lstm_36/lstm_cell_36/bias/m
&:$ 2Adam/dense_16/kernel/v
 :2Adam/dense_16/bias/v
3:1	?2"Adam/lstm_35/lstm_cell_35/kernel/v
=:;	@?2,Adam/lstm_35/lstm_cell_35/recurrent_kernel/v
-:+?2 Adam/lstm_35/lstm_cell_35/bias/v
3:1	@?2"Adam/lstm_36/lstm_cell_36/kernel/v
=:;	 ?2,Adam/lstm_36/lstm_cell_36/recurrent_kernel/v
-:+?2 Adam/lstm_36/lstm_cell_36/bias/v
?2?
.__inference_sequential_20_layer_call_fn_402942
.__inference_sequential_20_layer_call_fn_403670
.__inference_sequential_20_layer_call_fn_403691
.__inference_sequential_20_layer_call_fn_403464?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_20_layer_call_and_return_conditional_losses_404016
I__inference_sequential_20_layer_call_and_return_conditional_losses_404348
I__inference_sequential_20_layer_call_and_return_conditional_losses_403524
I__inference_sequential_20_layer_call_and_return_conditional_losses_403584?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_401676lstm_35_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_lstm_35_layer_call_fn_404377
(__inference_lstm_35_layer_call_fn_404388
(__inference_lstm_35_layer_call_fn_404399
(__inference_lstm_35_layer_call_fn_404410?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404571
C__inference_lstm_35_layer_call_and_return_conditional_losses_404732
C__inference_lstm_35_layer_call_and_return_conditional_losses_404893
C__inference_lstm_35_layer_call_and_return_conditional_losses_405054?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_lstm_36_layer_call_fn_405083
(__inference_lstm_36_layer_call_fn_405094
(__inference_lstm_36_layer_call_fn_405105
(__inference_lstm_36_layer_call_fn_405116?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405277
C__inference_lstm_36_layer_call_and_return_conditional_losses_405438
C__inference_lstm_36_layer_call_and_return_conditional_losses_405599
C__inference_lstm_36_layer_call_and_return_conditional_losses_405760?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_16_layer_call_fn_405765
+__inference_dropout_16_layer_call_fn_405770?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_16_layer_call_and_return_conditional_losses_405775
F__inference_dropout_16_layer_call_and_return_conditional_losses_405787?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_16_layer_call_fn_405796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_16_layer_call_and_return_conditional_losses_405806?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_403649lstm_35_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_lstm_cell_35_layer_call_fn_405841
-__inference_lstm_cell_35_layer_call_fn_405858?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405908
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405958?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_0_405969?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_405980?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_405991?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
-__inference_lstm_cell_36_layer_call_fn_406026
-__inference_lstm_cell_36_layer_call_fn_406043?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406093
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406143?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_3_406154?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_406165?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_406176?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
!__inference__wrapped_model_401676{&'()*+:?7
0?-
+?(
lstm_35_input?????????
? "3?0
.
dense_16"?
dense_16??????????
D__inference_dense_16_layer_call_and_return_conditional_losses_405806\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_16_layer_call_fn_405796O/?,
%?"
 ?
inputs????????? 
? "???????????
F__inference_dropout_16_layer_call_and_return_conditional_losses_405775\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
F__inference_dropout_16_layer_call_and_return_conditional_losses_405787\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ~
+__inference_dropout_16_layer_call_fn_405765O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ~
+__inference_dropout_16_layer_call_fn_405770O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ;
__inference_loss_fn_0_405969&?

? 
? "? ;
__inference_loss_fn_1_405980'?

? 
? "? ;
__inference_loss_fn_2_405991(?

? 
? "? ;
__inference_loss_fn_3_406154)?

? 
? "? ;
__inference_loss_fn_4_406165*?

? 
? "? ;
__inference_loss_fn_5_406176+?

? 
? "? ?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404571?&'(O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????@
? ?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404732?&'(O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????@
? ?
C__inference_lstm_35_layer_call_and_return_conditional_losses_404893q&'(??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????@
? ?
C__inference_lstm_35_layer_call_and_return_conditional_losses_405054q&'(??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????@
? ?
(__inference_lstm_35_layer_call_fn_404377}&'(O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"??????????????????@?
(__inference_lstm_35_layer_call_fn_404388}&'(O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"??????????????????@?
(__inference_lstm_35_layer_call_fn_404399d&'(??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????@?
(__inference_lstm_35_layer_call_fn_404410d&'(??<
5?2
$?!
inputs?????????

 
p

 
? "??????????@?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405277})*+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "%?"
?
0????????? 
? ?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405438})*+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "%?"
?
0????????? 
? ?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405599m)*+??<
5?2
$?!
inputs?????????@

 
p 

 
? "%?"
?
0????????? 
? ?
C__inference_lstm_36_layer_call_and_return_conditional_losses_405760m)*+??<
5?2
$?!
inputs?????????@

 
p

 
? "%?"
?
0????????? 
? ?
(__inference_lstm_36_layer_call_fn_405083p)*+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "?????????? ?
(__inference_lstm_36_layer_call_fn_405094p)*+O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "?????????? ?
(__inference_lstm_36_layer_call_fn_405105`)*+??<
5?2
$?!
inputs?????????@

 
p 

 
? "?????????? ?
(__inference_lstm_36_layer_call_fn_405116`)*+??<
5?2
$?!
inputs?????????@

 
p

 
? "?????????? ?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405908?&'(??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
H__inference_lstm_cell_35_layer_call_and_return_conditional_losses_405958?&'(??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
-__inference_lstm_cell_35_layer_call_fn_405841?&'(??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
-__inference_lstm_cell_35_layer_call_fn_405858?&'(??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406093?)*+??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_406143?)*+??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
-__inference_lstm_cell_36_layer_call_fn_406026?)*+??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
-__inference_lstm_cell_36_layer_call_fn_406043?)*+??}
v?s
 ?
inputs?????????@
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
I__inference_sequential_20_layer_call_and_return_conditional_losses_403524u&'()*+B??
8?5
+?(
lstm_35_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_20_layer_call_and_return_conditional_losses_403584u&'()*+B??
8?5
+?(
lstm_35_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_20_layer_call_and_return_conditional_losses_404016n&'()*+;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_20_layer_call_and_return_conditional_losses_404348n&'()*+;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_20_layer_call_fn_402942h&'()*+B??
8?5
+?(
lstm_35_input?????????
p 

 
? "???????????
.__inference_sequential_20_layer_call_fn_403464h&'()*+B??
8?5
+?(
lstm_35_input?????????
p

 
? "???????????
.__inference_sequential_20_layer_call_fn_403670a&'()*+;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
.__inference_sequential_20_layer_call_fn_403691a&'()*+;?8
1?.
$?!
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_403649?&'()*+K?H
? 
A?>
<
lstm_35_input+?(
lstm_35_input?????????"3?0
.
dense_16"?
dense_16?????????