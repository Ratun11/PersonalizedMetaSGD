ь§
ц
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ыи
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
n
dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense3/bias
g
dense3/bias/Read/ReadVariableOpReadVariableOpdense3/bias*
_output_shapes
:
*
dtype0
w
dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense3/kernel
p
!dense3/kernel/Read/ReadVariableOpReadVariableOpdense3/kernel*
_output_shapes
:	
*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
*
dtype0
l

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
conv3/bias
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:P*
dtype0
}
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv3/kernel
v
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*'
_output_shapes
:P*
dtype0
m

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
f
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes	
:*
dtype0
~
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
w
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*(
_output_shapes
:*
dtype0

serving_default_reshape_6_inputPlaceholder*,
_output_shapes
:џџџџџџџџџ*
dtype0*!
shape:џџџџџџџџџ
З
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_6_inputconv1/kernel
conv1/biasconv3/kernel
conv3/biasdense1/kerneldense1/biasdense3/kerneldense3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1891681

NoOpNoOp
іH
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*БH
valueЇHBЄH BH
в
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
Ш
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*
Ѕ
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator* 

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
Ш
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op*
Ѕ
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator* 

H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
І
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias*
Ѕ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator* 
І
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias*

e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
<
(0
)1
>2
?3
T4
U5
c6
d7*
<
(0
)1
>2
?3
T4
U5
c6
d7*
* 
А
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ptrace_0
qtrace_1* 

rtrace_0
strace_1* 
* 
O
t
_variables
u_iterations
v_learning_rate
w_update_step_xla*

xserving_default* 
* 
* 
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

~trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

(0
)1*

(0
)1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

>0
?1*

>0
?1*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

Ѓtrace_0* 

Єtrace_0* 
\V
VARIABLE_VALUEconv3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

Њtrace_0
Ћtrace_1* 

Ќtrace_0
­trace_1* 
* 
* 
* 
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 

T0
U1*

T0
U1*
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

Сtrace_0
Тtrace_1* 

Уtrace_0
Фtrace_1* 
* 

c0
d1*

c0
d1*
* 

Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

Ъtrace_0* 

Ыtrace_0* 
]W
VARIABLE_VALUEdense3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

бtrace_0* 

вtrace_0* 
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*

г0
д1*
* 
* 
* 
* 
* 
* 

u0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
е	variables
ж	keras_api

зtotal

иcount*
M
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs*

з0
и1*

е	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1*

й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
а
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv3/kernel
conv3/biasdense1/kerneldense1/biasdense3/kerneldense3/bias	iterationlearning_ratetotal_1count_1totalcountConst*
Tin
2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1892017
Ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv3/kernel
conv3/biasdense1/kerneldense1/biasdense3/kerneldense3/bias	iterationlearning_ratetotal_1count_1totalcount*
Tin
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1892068ры
ы
i
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891769

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

ї
C__inference_dense1_layer_call_and_return_conditional_losses_1891438

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
e
,__inference_dropout_11_layer_call_fn_1891852

inputs
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891455p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№	
b
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891911

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ё

(__inference_dense3_layer_call_fn_1891883

inputs
unknown:	

	unknown_0:

identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_1891467o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891879:'#
!
_user_specified_name	1891877:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891827

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ш
b
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891359

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
о
.__inference_sequential_3_layer_call_fn_1891576
reshape_6_input#
unknown:
	unknown_0:	$
	unknown_1:P
	unknown_2:P
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallreshape_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891534o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891572:'#
!
_user_specified_name	1891570:'#
!
_user_specified_name	1891568:'#
!
_user_specified_name	1891566:'#
!
_user_specified_name	1891564:'#
!
_user_specified_name	1891562:'#
!
_user_specified_name	1891560:'#
!
_user_specified_name	1891558:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
ы
i
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891327

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
г

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891419

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџPQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџP*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџPT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџPi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ы
i
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891338

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
ў
B__inference_conv1_layer_call_and_return_conditional_losses_1891372

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
H
,__inference_dropout_11_layer_call_fn_1891857

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891525a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


'__inference_conv3_layer_call_fn_1891778

inputs"
unknown:P
	unknown_0:P
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv3_layer_call_and_return_conditional_losses_1891402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891774:'#
!
_user_specified_name	1891772:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
i
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891711

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
е
%__inference_signature_wrapper_1891681
reshape_6_input#
unknown:
	unknown_0:	$
	unknown_1:P
	unknown_2:P
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallreshape_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1891321o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891677:'#
!
_user_specified_name	1891675:'#
!
_user_specified_name	1891673:'#
!
_user_specified_name	1891671:'#
!
_user_specified_name	1891669:'#
!
_user_specified_name	1891667:'#
!
_user_specified_name	1891665:'#
!
_user_specified_name	1891663:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
§3
ѕ
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891487
reshape_6_input)
conv1_1891373:
conv1_1891375:	(
conv3_1891403:P
conv3_1891405:P"
dense1_1891439:

dense1_1891441:	!
dense3_1891468:	

dense3_1891470:

identityЂconv1/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂdense3/StatefulPartitionedCallЂ"dropout_10/StatefulPartitionedCallЂ"dropout_11/StatefulPartitionedCallЂ!dropout_9/StatefulPartitionedCallЭ
reshape_6/PartitionedCallPartitionedCallreshape_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891359ю
 zero_padding2d_6/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891327
conv1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_6/PartitionedCall:output:0conv1_1891373conv1_1891375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_1891372є
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891389і
 zero_padding2d_7/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891338
conv3/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_7/PartitionedCall:output:0conv3_1891403conv3_1891405*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv3_layer_call_and_return_conditional_losses_1891402
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891419с
flatten_3/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891426
dense1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense1_1891439dense1_1891441*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_1891438
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891455
dense3/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense3_1891468dense3_1891470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_1891467м
reshape_7/PartitionedCallPartitionedCall'dense3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891484q
IdentityIdentity"reshape_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^conv1/StatefulPartitionedCall^conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense3/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:'#
!
_user_specified_name	1891470:'#
!
_user_specified_name	1891468:'#
!
_user_specified_name	1891441:'#
!
_user_specified_name	1891439:'#
!
_user_specified_name	1891405:'#
!
_user_specified_name	1891403:'#
!
_user_specified_name	1891375:'#
!
_user_specified_name	1891373:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
ЂQ
т
"__inference__wrapped_model_1891321
reshape_6_inputM
1sequential_3_conv1_conv2d_readvariableop_resource:A
2sequential_3_conv1_biasadd_readvariableop_resource:	L
1sequential_3_conv3_conv2d_readvariableop_resource:P@
2sequential_3_conv3_biasadd_readvariableop_resource:PF
2sequential_3_dense1_matmul_readvariableop_resource:
B
3sequential_3_dense1_biasadd_readvariableop_resource:	E
2sequential_3_dense3_matmul_readvariableop_resource:	
A
3sequential_3_dense3_biasadd_readvariableop_resource:

identityЂ)sequential_3/conv1/BiasAdd/ReadVariableOpЂ(sequential_3/conv1/Conv2D/ReadVariableOpЂ)sequential_3/conv3/BiasAdd/ReadVariableOpЂ(sequential_3/conv3/Conv2D/ReadVariableOpЂ*sequential_3/dense1/BiasAdd/ReadVariableOpЂ)sequential_3/dense1/MatMul/ReadVariableOpЂ*sequential_3/dense3/BiasAdd/ReadVariableOpЂ)sequential_3/dense3/MatMul/ReadVariableOpi
sequential_3/reshape_6/ShapeShapereshape_6_input*
T0*
_output_shapes
::эЯt
*sequential_3/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_3/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_3/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$sequential_3/reshape_6/strided_sliceStridedSlice%sequential_3/reshape_6/Shape:output:03sequential_3/reshape_6/strided_slice/stack:output:05sequential_3/reshape_6/strided_slice/stack_1:output:05sequential_3/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_3/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&sequential_3/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :i
&sequential_3/reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :
$sequential_3/reshape_6/Reshape/shapePack-sequential_3/reshape_6/strided_slice:output:0/sequential_3/reshape_6/Reshape/shape/1:output:0/sequential_3/reshape_6/Reshape/shape/2:output:0/sequential_3/reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Є
sequential_3/reshape_6/ReshapeReshapereshape_6_input-sequential_3/reshape_6/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
*sequential_3/zero_padding2d_6/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               С
!sequential_3/zero_padding2d_6/PadPad'sequential_3/reshape_6/Reshape:output:03sequential_3/zero_padding2d_6/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџЄ
(sequential_3/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0х
sequential_3/conv1/Conv2DConv2D*sequential_3/zero_padding2d_6/Pad:output:00sequential_3/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides

)sequential_3/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0З
sequential_3/conv1/BiasAddBiasAdd"sequential_3/conv1/Conv2D:output:01sequential_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/conv1/ReluRelu#sequential_3/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/dropout_9/IdentityIdentity%sequential_3/conv1/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ
*sequential_3/zero_padding2d_7/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               Т
!sequential_3/zero_padding2d_7/PadPad(sequential_3/dropout_9/Identity:output:03sequential_3/zero_padding2d_7/Pad/paddings:output:0*
T0*0
_output_shapes
:џџџџџџџџџЃ
(sequential_3/conv3/Conv2D/ReadVariableOpReadVariableOp1sequential_3_conv3_conv2d_readvariableop_resource*'
_output_shapes
:P*
dtype0ф
sequential_3/conv3/Conv2DConv2D*sequential_3/zero_padding2d_7/Pad:output:00sequential_3/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџP*
paddingVALID*
strides

)sequential_3/conv3/BiasAdd/ReadVariableOpReadVariableOp2sequential_3_conv3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0Ж
sequential_3/conv3/BiasAddBiasAdd"sequential_3/conv3/Conv2D:output:01sequential_3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџP~
sequential_3/conv3/ReluRelu#sequential_3/conv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџP
 sequential_3/dropout_10/IdentityIdentity%sequential_3/conv3/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџPm
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  Ў
sequential_3/flatten_3/ReshapeReshape)sequential_3/dropout_10/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
)sequential_3/dense1/MatMul/ReadVariableOpReadVariableOp2sequential_3_dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Г
sequential_3/dense1/MatMulMatMul'sequential_3/flatten_3/Reshape:output:01sequential_3/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
*sequential_3/dense1/BiasAdd/ReadVariableOpReadVariableOp3sequential_3_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Г
sequential_3/dense1/BiasAddBiasAdd$sequential_3/dense1/MatMul:product:02sequential_3/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
sequential_3/dense1/ReluRelu$sequential_3/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
 sequential_3/dropout_11/IdentityIdentity&sequential_3/dense1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ
)sequential_3/dense3/MatMul/ReadVariableOpReadVariableOp2sequential_3_dense3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0Д
sequential_3/dense3/MatMulMatMul)sequential_3/dropout_11/Identity:output:01sequential_3/dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

*sequential_3/dense3/BiasAdd/ReadVariableOpReadVariableOp3sequential_3_dense3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0В
sequential_3/dense3/BiasAddBiasAdd$sequential_3/dense3/MatMul:product:02sequential_3/dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
~
sequential_3/dense3/SoftmaxSoftmax$sequential_3/dense3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

sequential_3/reshape_7/ShapeShape%sequential_3/dense3/Softmax:softmax:0*
T0*
_output_shapes
::эЯt
*sequential_3/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_3/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_3/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$sequential_3/reshape_7/strided_sliceStridedSlice%sequential_3/reshape_7/Shape:output:03sequential_3/reshape_7/strided_slice/stack:output:05sequential_3/reshape_7/strided_slice/stack_1:output:05sequential_3/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_3/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
К
$sequential_3/reshape_7/Reshape/shapePack-sequential_3/reshape_7/strided_slice:output:0/sequential_3/reshape_7/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
sequential_3/reshape_7/ReshapeReshape%sequential_3/dense3/Softmax:softmax:0-sequential_3/reshape_7/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
v
IdentityIdentity'sequential_3/reshape_7/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp*^sequential_3/conv1/BiasAdd/ReadVariableOp)^sequential_3/conv1/Conv2D/ReadVariableOp*^sequential_3/conv3/BiasAdd/ReadVariableOp)^sequential_3/conv3/Conv2D/ReadVariableOp+^sequential_3/dense1/BiasAdd/ReadVariableOp*^sequential_3/dense1/MatMul/ReadVariableOp+^sequential_3/dense3/BiasAdd/ReadVariableOp*^sequential_3/dense3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 2V
)sequential_3/conv1/BiasAdd/ReadVariableOp)sequential_3/conv1/BiasAdd/ReadVariableOp2T
(sequential_3/conv1/Conv2D/ReadVariableOp(sequential_3/conv1/Conv2D/ReadVariableOp2V
)sequential_3/conv3/BiasAdd/ReadVariableOp)sequential_3/conv3/BiasAdd/ReadVariableOp2T
(sequential_3/conv3/Conv2D/ReadVariableOp(sequential_3/conv3/Conv2D/ReadVariableOp2X
*sequential_3/dense1/BiasAdd/ReadVariableOp*sequential_3/dense1/BiasAdd/ReadVariableOp2V
)sequential_3/dense1/MatMul/ReadVariableOp)sequential_3/dense1/MatMul/ReadVariableOp2X
*sequential_3/dense3/BiasAdd/ReadVariableOp*sequential_3/dense3/BiasAdd/ReadVariableOp2V
)sequential_3/dense3/MatMul/ReadVariableOp)sequential_3/dense3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
М
N
2__inference_zero_padding2d_6_layer_call_fn_1891705

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891327
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891455

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891513

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџPc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџP"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ё
e
,__inference_dropout_10_layer_call_fn_1891794

inputs
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891419w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџP<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Д/

I__inference_sequential_3_layer_call_and_return_conditional_losses_1891534
reshape_6_input)
conv1_1891492:
conv1_1891494:	(
conv3_1891504:P
conv3_1891506:P"
dense1_1891516:

dense1_1891518:	!
dense3_1891527:	

dense3_1891529:

identityЂconv1/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂdense3/StatefulPartitionedCallЭ
reshape_6/PartitionedCallPartitionedCallreshape_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891359ю
 zero_padding2d_6/PartitionedCallPartitionedCall"reshape_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891327
conv1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_6/PartitionedCall:output:0conv1_1891492conv1_1891494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_1891372ф
dropout_9/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891501ю
 zero_padding2d_7/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891338
conv3/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_7/PartitionedCall:output:0conv3_1891504conv3_1891506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv3_layer_call_and_return_conditional_losses_1891402х
dropout_10/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891513й
flatten_3/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891426
dense1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense1_1891516dense1_1891518*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_1891438п
dropout_11/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891525
dense3/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense3_1891527dense3_1891529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_1891467м
reshape_7/PartitionedCallPartitionedCall'dense3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891484q
IdentityIdentity"reshape_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Є
NoOpNoOp^conv1/StatefulPartitionedCall^conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall:'#
!
_user_specified_name	1891529:'#
!
_user_specified_name	1891527:'#
!
_user_specified_name	1891518:'#
!
_user_specified_name	1891516:'#
!
_user_specified_name	1891506:'#
!
_user_specified_name	1891504:'#
!
_user_specified_name	1891494:'#
!
_user_specified_name	1891492:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
о
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891874

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
d
+__inference_dropout_9_layer_call_fn_1891736

inputs
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891389x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
о
.__inference_sequential_3_layer_call_fn_1891555
reshape_6_input#
unknown:
	unknown_0:	$
	unknown_1:P
	unknown_2:P
	unknown_3:

	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallreshape_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891551:'#
!
_user_specified_name	1891549:'#
!
_user_specified_name	1891547:'#
!
_user_specified_name	1891545:'#
!
_user_specified_name	1891543:'#
!
_user_specified_name	1891541:'#
!
_user_specified_name	1891539:'#
!
_user_specified_name	1891537:] Y
,
_output_shapes
:џџџџџџџџџ
)
_user_specified_namereshape_6_input
У
H
,__inference_dropout_10_layer_call_fn_1891799

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891513h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Х
G
+__inference_dropout_9_layer_call_fn_1891741

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891501i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕ

(__inference_dense1_layer_call_fn_1891836

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_1891438p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891832:'#
!
_user_specified_name	1891830:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891758

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О
ў
B__inference_conv1_layer_call_and_return_conditional_losses_1891731

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891816

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџPc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџP"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Ж
ќ
B__inference_conv3_layer_call_and_return_conditional_losses_1891789

inputs9
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:P*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџP*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџPX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџPi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџPS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891501

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
УB
ј
#__inference__traced_restore_1892068
file_prefix9
assignvariableop_conv1_kernel:,
assignvariableop_1_conv1_bias:	:
assignvariableop_2_conv3_kernel:P+
assignvariableop_3_conv3_bias:P4
 assignvariableop_4_dense1_kernel:
-
assignvariableop_5_dense1_bias:	3
 assignvariableop_6_dense3_kernel:	
,
assignvariableop_7_dense3_bias:
&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: 
identity_15ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Њ
value BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: Ь
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:+'
%
_user_specified_namedense3/bias:-)
'
_user_specified_namedense3/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:*&
$
_user_specified_name
conv3/bias:,(
&
_user_specified_nameconv3/kernel:*&
$
_user_specified_name
conv1/bias:,(
&
_user_specified_nameconv1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ж

ї
C__inference_dense1_layer_call_and_return_conditional_losses_1891847

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№	
b
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891484

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
й

e
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891753

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Џ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891525

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
G
+__inference_flatten_3_layer_call_fn_1891821

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891426a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
г

ѕ
C__inference_dense3_layer_call_and_return_conditional_losses_1891467

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
ќ
B__inference_conv3_layer_call_and_return_conditional_losses_1891402

inputs9
conv2d_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:P*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџP*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџPX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџPi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџPS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891811

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџPQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџP*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџPT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџPi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџP"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ыq

 __inference__traced_save_1892017
file_prefix?
#read_disablecopyonread_conv1_kernel:2
#read_1_disablecopyonread_conv1_bias:	@
%read_2_disablecopyonread_conv3_kernel:P1
#read_3_disablecopyonread_conv3_bias:P:
&read_4_disablecopyonread_dense1_kernel:
3
$read_5_disablecopyonread_dense1_bias:	9
&read_6_disablecopyonread_dense3_kernel:	
2
$read_7_disablecopyonread_dense3_bias:
,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: +
!read_10_disablecopyonread_total_1: +
!read_11_disablecopyonread_count_1: )
read_12_disablecopyonread_total: )
read_13_disablecopyonread_count: 
savev2_const
identity_29ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_conv1_kernel"/device:CPU:0*
_output_shapes
 Љ
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_conv1_kernel^Read/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:*
dtype0s
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:k

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*(
_output_shapes
:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_conv1_bias"/device:CPU:0*
_output_shapes
  
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_conv1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_conv3_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_conv3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:P*
dtype0v

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:Pl

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*'
_output_shapes
:Pw
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_conv3_bias"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_conv3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:P*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:P_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:Pz
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 Ё
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_dense1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense3_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	
x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_dense3_bias"/device:CPU:0*
_output_shapes
  
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_dense3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_10/DisableCopyOnReadDisableCopyOnRead!read_10_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOp!read_10_disablecopyonread_total_1^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_11/DisableCopyOnReadDisableCopyOnRead!read_11_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_11/ReadVariableOpReadVariableOp!read_11_disablecopyonread_count_1^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_12/DisableCopyOnReadDisableCopyOnReadread_12_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_12/ReadVariableOpReadVariableOpread_12_disablecopyonread_total^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_13/DisableCopyOnReadDisableCopyOnReadread_13_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_13/ReadVariableOpReadVariableOpread_13_disablecopyonread_count^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Њ
value BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:+'
%
_user_specified_namedense3/bias:-)
'
_user_specified_namedense3/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:*&
$
_user_specified_name
conv3/bias:,(
&
_user_specified_nameconv3/kernel:*&
$
_user_specified_name
conv1/bias:,(
&
_user_specified_nameconv1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ш
b
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891700

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891869

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


'__inference_conv1_layer_call_fn_1891720

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_1891372x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1891716:'#
!
_user_specified_name	1891714:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й

e
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891389

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Џ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г

ѕ
C__inference_dense3_layer_call_and_return_conditional_losses_1891894

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
N
2__inference_zero_padding2d_7_layer_call_fn_1891763

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891338
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
G
+__inference_reshape_6_layer_call_fn_1891686

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891359i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891426

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџP:W S
/
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Ё
G
+__inference_reshape_7_layer_call_fn_1891899

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891484`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs"ЇL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
serving_default­
P
reshape_6_input=
!serving_default_reshape_6_input:0џџџџџџџџџ=
	reshape_70
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:ь
ь
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
н
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
М
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator"
_tf_keras_layer
Ѕ
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
н
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
М
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer
Ѕ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias"
_tf_keras_layer
М
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator"
_tf_keras_layer
Л
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias"
_tf_keras_layer
Ѕ
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
X
(0
)1
>2
?3
T4
U5
c6
d7"
trackable_list_wrapper
X
(0
)1
>2
?3
T4
U5
c6
d7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
ptrace_0
qtrace_12
.__inference_sequential_3_layer_call_fn_1891555
.__inference_sequential_3_layer_call_fn_1891576Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zptrace_0zqtrace_1

rtrace_0
strace_12Ю
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891487
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891534Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zrtrace_0zstrace_1
еBв
"__inference__wrapped_model_1891321reshape_6_input"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
j
t
_variables
u_iterations
v_learning_rate
w_update_step_xla"
experimentalOptimizer
,
xserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
х
~trace_02Ш
+__inference_reshape_6_layer_call_fn_1891686
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0

trace_02у
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891700
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
2__inference_zero_padding2d_6_layer_call_fn_1891705
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891711
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
у
trace_02Ф
'__inference_conv1_layer_call_fn_1891720
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ў
trace_02п
B__inference_conv1_layer_call_and_return_conditional_losses_1891731
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2conv1/kernel
:2
conv1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
С
trace_0
trace_12
+__inference_dropout_9_layer_call_fn_1891736
+__inference_dropout_9_layer_call_fn_1891741Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ї
trace_0
trace_12М
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891753
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891758Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
2__inference_zero_padding2d_7_layer_call_fn_1891763
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891769
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
у
Ѓtrace_02Ф
'__inference_conv3_layer_call_fn_1891778
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0
ў
Єtrace_02п
B__inference_conv3_layer_call_and_return_conditional_losses_1891789
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0
':%P2conv3/kernel
:P2
conv3/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
У
Њtrace_0
Ћtrace_12
,__inference_dropout_10_layer_call_fn_1891794
,__inference_dropout_10_layer_call_fn_1891799Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0zЋtrace_1
љ
Ќtrace_0
­trace_12О
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891811
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891816Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ч
Гtrace_02Ш
+__inference_flatten_3_layer_call_fn_1891821
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0

Дtrace_02у
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891827
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
ф
Кtrace_02Х
(__inference_dense1_layer_call_fn_1891836
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zКtrace_0
џ
Лtrace_02р
C__inference_dense1_layer_call_and_return_conditional_losses_1891847
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0
!:
2dense1/kernel
:2dense1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
У
Сtrace_0
Тtrace_12
,__inference_dropout_11_layer_call_fn_1891852
,__inference_dropout_11_layer_call_fn_1891857Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zСtrace_0zТtrace_1
љ
Уtrace_0
Фtrace_12О
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891869
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891874Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0zФtrace_1
"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
ф
Ъtrace_02Х
(__inference_dense3_layer_call_fn_1891883
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0
џ
Ыtrace_02р
C__inference_dense3_layer_call_and_return_conditional_losses_1891894
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0
 :	
2dense3/kernel
:
2dense3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ч
бtrace_02Ш
+__inference_reshape_7_layer_call_fn_1891899
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0

вtrace_02у
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891911
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѕBђ
.__inference_sequential_3_layer_call_fn_1891555reshape_6_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
.__inference_sequential_3_layer_call_fn_1891576reshape_6_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891487reshape_6_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891534reshape_6_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
'
u0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
сBо
%__inference_signature_wrapper_1891681reshape_6_input"Ё
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 $

kwonlyargs
jreshape_6_input
kwonlydefaults
 
annotationsЊ *
 
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
еBв
+__inference_reshape_6_layer_call_fn_1891686inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891700inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
мBй
2__inference_zero_padding2d_6_layer_call_fn_1891705inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891711inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
бBЮ
'__inference_conv1_layer_call_fn_1891720inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv1_layer_call_and_return_conditional_losses_1891731inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
сBо
+__inference_dropout_9_layer_call_fn_1891736inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
сBо
+__inference_dropout_9_layer_call_fn_1891741inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891753inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891758inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
мBй
2__inference_zero_padding2d_7_layer_call_fn_1891763inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891769inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
бBЮ
'__inference_conv3_layer_call_fn_1891778inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_conv3_layer_call_and_return_conditional_losses_1891789inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
тBп
,__inference_dropout_10_layer_call_fn_1891794inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
тBп
,__inference_dropout_10_layer_call_fn_1891799inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891811inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891816inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
еBв
+__inference_flatten_3_layer_call_fn_1891821inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891827inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
вBЯ
(__inference_dense1_layer_call_fn_1891836inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense1_layer_call_and_return_conditional_losses_1891847inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
тBп
,__inference_dropout_11_layer_call_fn_1891852inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
тBп
,__inference_dropout_11_layer_call_fn_1891857inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891869inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891874inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
вBЯ
(__inference_dense3_layer_call_fn_1891883inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense3_layer_call_and_return_conditional_losses_1891894inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
еBв
+__inference_reshape_7_layer_call_fn_1891899inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891911inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
е	variables
ж	keras_api

зtotal

иcount"
_tf_keras_metric
c
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs"
_tf_keras_metric
0
з0
и1"
trackable_list_wrapper
.
е	variables"
_generic_user_object
:  (2total
:  (2count
0
л0
м1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЇ
"__inference__wrapped_model_1891321()>?TUcd=Ђ:
3Ђ0
.+
reshape_6_inputџџџџџџџџџ
Њ "5Њ2
0
	reshape_7# 
	reshape_7џџџџџџџџџ
Л
B__inference_conv1_layer_call_and_return_conditional_losses_1891731u()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
'__inference_conv1_layer_call_fn_1891720j()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџК
B__inference_conv3_layer_call_and_return_conditional_losses_1891789t>?8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџP
 
'__inference_conv3_layer_call_fn_1891778i>?8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџPЌ
C__inference_dense1_layer_call_and_return_conditional_losses_1891847eTU0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
(__inference_dense1_layer_call_fn_1891836ZTU0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџЋ
C__inference_dense3_layer_call_and_return_conditional_losses_1891894dcd0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
(__inference_dense3_layer_call_fn_1891883Ycd0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
О
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891811s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџP
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџP
 О
G__inference_dropout_10_layer_call_and_return_conditional_losses_1891816s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџP
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџP
 
,__inference_dropout_10_layer_call_fn_1891794h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџP
p
Њ ")&
unknownџџџџџџџџџP
,__inference_dropout_10_layer_call_fn_1891799h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџP
p 
Њ ")&
unknownџџџџџџџџџPА
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891869e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 А
G__inference_dropout_11_layer_call_and_return_conditional_losses_1891874e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
,__inference_dropout_11_layer_call_fn_1891852Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџ
,__inference_dropout_11_layer_call_fn_1891857Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџП
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891753u<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 П
F__inference_dropout_9_layer_call_and_return_conditional_losses_1891758u<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
+__inference_dropout_9_layer_call_fn_1891736j<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "*'
unknownџџџџџџџџџ
+__inference_dropout_9_layer_call_fn_1891741j<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "*'
unknownџџџџџџџџџВ
F__inference_flatten_3_layer_call_and_return_conditional_losses_1891827h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџP
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
+__inference_flatten_3_layer_call_fn_1891821]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџP
Њ ""
unknownџџџџџџџџџЗ
F__inference_reshape_6_layer_call_and_return_conditional_losses_1891700m4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
+__inference_reshape_6_layer_call_fn_1891686b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџЉ
F__inference_reshape_7_layer_call_and_return_conditional_losses_1891911_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
+__inference_reshape_7_layer_call_fn_1891899T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "!
unknownџџџџџџџџџ
Ь
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891487()>?TUcdEЂB
;Ђ8
.+
reshape_6_inputџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 Ь
I__inference_sequential_3_layer_call_and_return_conditional_losses_1891534()>?TUcdEЂB
;Ђ8
.+
reshape_6_inputџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 І
.__inference_sequential_3_layer_call_fn_1891555t()>?TUcdEЂB
;Ђ8
.+
reshape_6_inputџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ
І
.__inference_sequential_3_layer_call_fn_1891576t()>?TUcdEЂB
;Ђ8
.+
reshape_6_inputџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ
Н
%__inference_signature_wrapper_1891681()>?TUcdPЂM
Ђ 
FЊC
A
reshape_6_input.+
reshape_6_inputџџџџџџџџџ"5Њ2
0
	reshape_7# 
	reshape_7џџџџџџџџџ
ї
M__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_1891711ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_zero_padding2d_6_layer_call_fn_1891705RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_1891769ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_zero_padding2d_7_layer_call_fn_1891763RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ