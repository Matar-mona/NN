       £K"	   fЊ÷Abrain.Event:2`*т∆ƒ     сОЂ:	#[6fЊ÷A"єЙ
Е
conv2d_1_inputPlaceholder*
dtype0*1
_output_shapes
:€€€€€€€€€АА*&
shape:€€€€€€€€€АА
v
conv2d_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *0Њ*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *0>*
dtype0*
_output_shapes
: 
≤
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2ОЏЊ*
seed±€е)
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:
Й
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:*
T0
У
conv2d_1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
»
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
о
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ш
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА
c
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*1
_output_shapes
:€€€€€€€€€АА
h
batch_normalization_1/ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
З
batch_normalization_1/gamma
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
д
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gammabatch_normalization_1/Const*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
j
batch_normalization_1/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    
Ж
batch_normalization_1/beta
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
г
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/betabatch_normalization_1/Const_1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:*
use_locking(
Ы
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
j
batch_normalization_1/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:
Н
!batch_normalization_1/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ш
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:
∞
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
j
batch_normalization_1/Const_3Const*
valueB*  А?*
dtype0*
_output_shapes
:
С
%batch_normalization_1/moving_variance
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
Д
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:
Љ
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
Й
4batch_normalization_1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
љ
"batch_normalization_1/moments/meanMeanconv2d_1/Relu4batch_normalization_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
П
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
T0*&
_output_shapes
:
ї
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/Relu*batch_normalization_1/moments/StopGradient*
T0*1
_output_shapes
:€€€€€€€€€АА
Н
8batch_normalization_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_1/moments/varianceMean/batch_normalization_1/moments/SquaredDifference8batch_normalization_1/moments/variance/reduction_indices*
T0*&
_output_shapes
:*

Tidx0*
	keep_dims(
Т
%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:
Ш
'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
T0*
_output_shapes
:*
squeeze_dims
 
j
%batch_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:
Я
#batch_normalization_1/batchnorm/addAdd'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes
:
x
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
_output_shapes
:*
T0
Ш
#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt batch_normalization_1/gamma/read*
T0*
_output_shapes
:
Ь
%batch_normalization_1/batchnorm/mul_1Mulconv2d_1/Relu#batch_normalization_1/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
Э
%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
T0*
_output_shapes
:
Ч
#batch_normalization_1/batchnorm/subSubbatch_normalization_1/beta/read%batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes
:
і
%batch_normalization_1/batchnorm/add_1Add%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
¶
+batch_normalization_1/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_1/AssignMovingAvg/subSub&batch_normalization_1/moving_mean/read%batch_normalization_1/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
г
)batch_normalization_1/AssignMovingAvg/mulMul)batch_normalization_1/AssignMovingAvg/sub+batch_normalization_1/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
о
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
ђ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_1/AssignMovingAvg_1/subSub*batch_normalization_1/moving_variance/read'batch_normalization_1/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
н
+batch_normalization_1/AssignMovingAvg_1/mulMul+batch_normalization_1/AssignMovingAvg_1/sub-batch_normalization_1/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
ъ
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:*
use_locking( 
o
*batch_normalization_1/keras_learning_phasePlaceholder*
dtype0
*
_output_shapes
:*
shape:
™
!batch_normalization_1/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_1/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
Л
#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Х
*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_1/cond/batchnorm/addAdd/batch_normalization_1/cond/batchnorm/add/Switch*batch_normalization_1/cond/batchnorm/add/y*
_output_shapes
:*
T0
о
/batch_normalization_1/cond/batchnorm/add/SwitchSwitch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
В
*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
T0*
_output_shapes
:
±
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt/batch_normalization_1/cond/batchnorm/mul/Switch*
_output_shapes
:*
T0
Џ
/batch_normalization_1/cond/batchnorm/mul/SwitchSwitch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_1/gamma
 
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
й
1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchconv2d_1/Relu"batch_normalization_1/cond/pred_id*
T0* 
_class
loc:@conv2d_1/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
≥
*batch_normalization_1/cond/batchnorm/mul_2Mul1batch_normalization_1/cond/batchnorm/mul_2/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*
_output_shapes
:
и
1batch_normalization_1/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
::
±
(batch_normalization_1/cond/batchnorm/subSub/batch_normalization_1/cond/batchnorm/sub/Switch*batch_normalization_1/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Ў
/batch_normalization_1/cond/batchnorm/sub/SwitchSwitchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_1/beta
√
*batch_normalization_1/cond/batchnorm/add_1Add*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
√
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
T0*
N*3
_output_shapes!
:€€€€€€€€€АА: 
v
conv2d_2/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
≤
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*&
_output_shapes
: *
seed2ееУ*
seed±€е)*
T0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
: *
T0
Й
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
: 
У
conv2d_2/kernel
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
»
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel
Ж
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: 
[
conv2d_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_2/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
≠
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
: *
T0* 
_class
loc:@conv2d_2/bias
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
conv2d_2/convolutionConv2D batch_normalization_1/cond/Mergeconv2d_2/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ *
	dilations

Ц
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@@ 
a
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@@ 
h
batch_normalization_2/ConstConst*
dtype0*
_output_shapes
: *
valueB *  А?
З
batch_normalization_2/gamma
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
д
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gammabatch_normalization_2/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
: 
Ю
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
j
batch_normalization_2/Const_1Const*
valueB *    *
dtype0*
_output_shapes
: 
Ж
batch_normalization_2/beta
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
г
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/betabatch_normalization_2/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
: 
Ы
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
j
batch_normalization_2/Const_2Const*
dtype0*
_output_shapes
: *
valueB *    
Н
!batch_normalization_2/moving_mean
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
ш
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
: 
∞
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
j
batch_normalization_2/Const_3Const*
valueB *  А?*
dtype0*
_output_shapes
: 
С
%batch_normalization_2/moving_variance
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Д
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
Љ
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
Й
4batch_normalization_2/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
љ
"batch_normalization_2/moments/meanMeanconv2d_2/Relu4batch_normalization_2/moments/mean/reduction_indices*
T0*&
_output_shapes
: *

Tidx0*
	keep_dims(
П
*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/mean*&
_output_shapes
: *
T0
є
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/Relu*batch_normalization_2/moments/StopGradient*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Н
8batch_normalization_2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_2/moments/varianceMean/batch_normalization_2/moments/SquaredDifference8batch_normalization_2/moments/variance/reduction_indices*&
_output_shapes
: *

Tidx0*
	keep_dims(*
T0
Т
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
: 
Ш
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
: 
j
%batch_normalization_2/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_2/batchnorm/addAdd'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
T0*
_output_shapes
: 
x
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
T0*
_output_shapes
: 
Ш
#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt batch_normalization_2/gamma/read*
T0*
_output_shapes
: 
Ъ
%batch_normalization_2/batchnorm/mul_1Mulconv2d_2/Relu#batch_normalization_2/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Э
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
T0*
_output_shapes
: 
Ч
#batch_normalization_2/batchnorm/subSubbatch_normalization_2/beta/read%batch_normalization_2/batchnorm/mul_2*
_output_shapes
: *
T0
≤
%batch_normalization_2/batchnorm/add_1Add%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
¶
+batch_normalization_2/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_2/moving_mean
Џ
)batch_normalization_2/AssignMovingAvg/subSub&batch_normalization_2/moving_mean/read%batch_normalization_2/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
г
)batch_normalization_2/AssignMovingAvg/mulMul)batch_normalization_2/AssignMovingAvg/sub+batch_normalization_2/AssignMovingAvg/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
о
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: *
use_locking( 
ђ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_2/moving_variance
ж
+batch_normalization_2/AssignMovingAvg_1/subSub*batch_normalization_2/moving_variance/read'batch_normalization_2/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
н
+batch_normalization_2/AssignMovingAvg_1/mulMul+batch_normalization_2/AssignMovingAvg_1/sub-batch_normalization_2/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
ъ
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
™
!batch_normalization_2/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_2/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
_output_shapes
:*
T0

З
#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Х
*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_2/cond/batchnorm/addAdd/batch_normalization_2/cond/batchnorm/add/Switch*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes
: 
о
/batch_normalization_2/cond/batchnorm/add/SwitchSwitch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
: : 
В
*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
T0*
_output_shapes
: 
±
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt/batch_normalization_2/cond/batchnorm/mul/Switch*
T0*
_output_shapes
: 
Џ
/batch_normalization_2/cond/batchnorm/mul/SwitchSwitch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
: : 
»
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
е
1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchconv2d_2/Relu"batch_normalization_2/cond/pred_id*
T0* 
_class
loc:@conv2d_2/Relu*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
≥
*batch_normalization_2/cond/batchnorm/mul_2Mul1batch_normalization_2/cond/batchnorm/mul_2/Switch(batch_normalization_2/cond/batchnorm/mul*
_output_shapes
: *
T0
и
1batch_normalization_2/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
: : 
±
(batch_normalization_2/cond/batchnorm/subSub/batch_normalization_2/cond/batchnorm/sub/Switch*batch_normalization_2/cond/batchnorm/mul_2*
T0*
_output_shapes
: 
Ў
/batch_normalization_2/cond/batchnorm/sub/SwitchSwitchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id* 
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_2/beta
Ѕ
*batch_normalization_2/cond/batchnorm/add_1Add*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Ѕ
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*
T0*
N*1
_output_shapes
:€€€€€€€€€@@ : 
v
conv2d_3/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
`
conv2d_3/random_uniform/maxConst*
valueB
 *  А=*
dtype0*
_output_shapes
: 
±
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
: @*
seed2∆ќ
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*&
_output_shapes
: @
Й
conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*&
_output_shapes
: @*
T0
У
conv2d_3/kernel
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
»
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel
Ж
conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
: @
[
conv2d_3/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_3/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
≠
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_3/bias/readIdentityconv2d_3/bias*
_output_shapes
:@*
T0* 
_class
loc:@conv2d_3/bias
s
"conv2d_3/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
conv2d_3/convolutionConv2D batch_normalization_2/cond/Mergeconv2d_3/kernel/read*/
_output_shapes
:€€€€€€€€€  @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ц
conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€  @
a
conv2d_3/ReluReluconv2d_3/BiasAdd*/
_output_shapes
:€€€€€€€€€  @*
T0
h
batch_normalization_3/ConstConst*
valueB@*  А?*
dtype0*
_output_shapes
:@
З
batch_normalization_3/gamma
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
д
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gammabatch_normalization_3/Const*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
Ю
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@
j
batch_normalization_3/Const_1Const*
valueB@*    *
dtype0*
_output_shapes
:@
Ж
batch_normalization_3/beta
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
г
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/betabatch_normalization_3/Const_1*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
Ы
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:@
j
batch_normalization_3/Const_2Const*
valueB@*    *
dtype0*
_output_shapes
:@
Н
!batch_normalization_3/moving_mean
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
ш
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(
∞
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
j
batch_normalization_3/Const_3Const*
valueB@*  А?*
dtype0*
_output_shapes
:@
С
%batch_normalization_3/moving_variance
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
Д
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Љ
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@
Й
4batch_normalization_3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
љ
"batch_normalization_3/moments/meanMeanconv2d_3/Relu4batch_normalization_3/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:@
П
*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/mean*
T0*&
_output_shapes
:@
є
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferenceconv2d_3/Relu*batch_normalization_3/moments/StopGradient*
T0*/
_output_shapes
:€€€€€€€€€  @
Н
8batch_normalization_3/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
з
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*&
_output_shapes
:@*

Tidx0*
	keep_dims(*
T0
Т
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:@
Ш
'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:@
j
%batch_normalization_3/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_3/batchnorm/addAdd'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
T0*
_output_shapes
:@
x
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes
:@
Ш
#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt batch_normalization_3/gamma/read*
_output_shapes
:@*
T0
Ъ
%batch_normalization_3/batchnorm/mul_1Mulconv2d_3/Relu#batch_normalization_3/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€  @
Э
%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
T0*
_output_shapes
:@
Ч
#batch_normalization_3/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_3/batchnorm/mul_2*
T0*
_output_shapes
:@
≤
%batch_normalization_3/batchnorm/add_1Add%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*/
_output_shapes
:€€€€€€€€€  @*
T0
¶
+batch_normalization_3/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_3/AssignMovingAvg/subSub&batch_normalization_3/moving_mean/read%batch_normalization_3/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
г
)batch_normalization_3/AssignMovingAvg/mulMul)batch_normalization_3/AssignMovingAvg/sub+batch_normalization_3/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
о
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
ђ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_3/AssignMovingAvg_1/subSub*batch_normalization_3/moving_variance/read'batch_normalization_3/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@
н
+batch_normalization_3/AssignMovingAvg_1/mulMul+batch_normalization_3/AssignMovingAvg_1/sub-batch_normalization_3/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@
ъ
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
_output_shapes
:@*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
™
!batch_normalization_3/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
_output_shapes

::*
T0

w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_3/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
_output_shapes
:*
T0

З
#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
Х
*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_3/cond/batchnorm/addAdd/batch_normalization_3/cond/batchnorm/add/Switch*batch_normalization_3/cond/batchnorm/add/y*
T0*
_output_shapes
:@
о
/batch_normalization_3/cond/batchnorm/add/SwitchSwitch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance* 
_output_shapes
:@:@
В
*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
T0*
_output_shapes
:@
±
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt/batch_normalization_3/cond/batchnorm/mul/Switch*
_output_shapes
:@*
T0
Џ
/batch_normalization_3/cond/batchnorm/mul/SwitchSwitch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:@:@
»
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€  @
е
1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchconv2d_3/Relu"batch_normalization_3/cond/pred_id*
T0* 
_class
loc:@conv2d_3/Relu*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
≥
*batch_normalization_3/cond/batchnorm/mul_2Mul1batch_normalization_3/cond/batchnorm/mul_2/Switch(batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:@*
T0
и
1batch_normalization_3/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id* 
_output_shapes
:@:@*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
±
(batch_normalization_3/cond/batchnorm/subSub/batch_normalization_3/cond/batchnorm/sub/Switch*batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:@*
T0
Ў
/batch_normalization_3/cond/batchnorm/sub/SwitchSwitchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id* 
_output_shapes
:@:@*
T0*-
_class#
!loc:@batch_normalization_3/beta
Ѕ
*batch_normalization_3/cond/batchnorm/add_1Add*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€  @
Ѕ
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*
T0*
N*1
_output_shapes
:€€€€€€€€€  @: 
u
up_sampling2d_1/ShapeShape batch_normalization_3/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
_output_shapes
:*
T0
ƒ
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_3/cond/Mergeup_sampling2d_1/mul*
align_corners( *
T0*/
_output_shapes
:€€€€€€€€€@@@
v
conv2d_4/random_uniform/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
`
conv2d_4/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А=
±
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
:@ *
seed2•Ь\
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*
T0*&
_output_shapes
:@ 
Й
conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*&
_output_shapes
:@ *
T0
У
conv2d_4/kernel
VariableV2*
shape:@ *
shared_name *
dtype0*&
_output_shapes
:@ *
	container 
»
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:@ 
Ж
conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@ 
[
conv2d_4/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
≠
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
: 
t
conv2d_4/bias/readIdentityconv2d_4/bias*
T0* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
s
"conv2d_4/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Г
conv2d_4/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ *
	dilations
*
T0
Ц
conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@@ 
a
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@@ 
h
batch_normalization_4/ConstConst*
dtype0*
_output_shapes
: *
valueB *  А?
З
batch_normalization_4/gamma
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
д
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gammabatch_normalization_4/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
: 
Ю
 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
j
batch_normalization_4/Const_1Const*
valueB *    *
dtype0*
_output_shapes
: 
Ж
batch_normalization_4/beta
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
г
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/betabatch_normalization_4/Const_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
: *
use_locking(
Ы
batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: 
j
batch_normalization_4/Const_2Const*
valueB *    *
dtype0*
_output_shapes
: 
Н
!batch_normalization_4/moving_mean
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ш
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes
: 
∞
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
j
batch_normalization_4/Const_3Const*
valueB *  А?*
dtype0*
_output_shapes
: 
С
%batch_normalization_4/moving_variance
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Д
,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes
: 
Љ
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
Й
4batch_normalization_4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_4/moments/meanMeanconv2d_4/Relu4batch_normalization_4/moments/mean/reduction_indices*
T0*&
_output_shapes
: *

Tidx0*
	keep_dims(
П
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
T0*&
_output_shapes
: 
є
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceconv2d_4/Relu*batch_normalization_4/moments/StopGradient*/
_output_shapes
:€€€€€€€€€@@ *
T0
Н
8batch_normalization_4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*&
_output_shapes
: *

Tidx0*
	keep_dims(*
T0
Т
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
T0*
_output_shapes
: *
squeeze_dims
 
Ш
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
: 
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_4/batchnorm/addAdd'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
T0*
_output_shapes
: 
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
_output_shapes
: *
T0
Ш
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_4/gamma/read*
T0*
_output_shapes
: 
Ъ
%batch_normalization_4/batchnorm/mul_1Mulconv2d_4/Relu#batch_normalization_4/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Э
%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
T0*
_output_shapes
: 
Ч
#batch_normalization_4/batchnorm/subSubbatch_normalization_4/beta/read%batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes
: 
≤
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
¶
+batch_normalization_4/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_4/AssignMovingAvg/subSub&batch_normalization_4/moving_mean/read%batch_normalization_4/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
г
)batch_normalization_4/AssignMovingAvg/mulMul)batch_normalization_4/AssignMovingAvg/sub+batch_normalization_4/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
о
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: *
use_locking( 
ђ
-batch_normalization_4/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_4/AssignMovingAvg_1/subSub*batch_normalization_4/moving_variance/read'batch_normalization_4/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
н
+batch_normalization_4/AssignMovingAvg_1/mulMul+batch_normalization_4/AssignMovingAvg_1/sub-batch_normalization_4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
ъ
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
™
!batch_normalization_4/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_4/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
З
#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Х
*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_4/cond/batchnorm/addAdd/batch_normalization_4/cond/batchnorm/add/Switch*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes
: 
о
/batch_normalization_4/cond/batchnorm/add/SwitchSwitch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id* 
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
В
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
T0*
_output_shapes
: 
±
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt/batch_normalization_4/cond/batchnorm/mul/Switch*
_output_shapes
: *
T0
Џ
/batch_normalization_4/cond/batchnorm/mul/SwitchSwitch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
: : 
»
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*/
_output_shapes
:€€€€€€€€€@@ *
T0
е
1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchconv2d_4/Relu"batch_normalization_4/cond/pred_id*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ *
T0* 
_class
loc:@conv2d_4/Relu
≥
*batch_normalization_4/cond/batchnorm/mul_2Mul1batch_normalization_4/cond/batchnorm/mul_2/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*
_output_shapes
: 
и
1batch_normalization_4/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean* 
_output_shapes
: : 
±
(batch_normalization_4/cond/batchnorm/subSub/batch_normalization_4/cond/batchnorm/sub/Switch*batch_normalization_4/cond/batchnorm/mul_2*
T0*
_output_shapes
: 
Ў
/batch_normalization_4/cond/batchnorm/sub/SwitchSwitchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
: : 
Ѕ
*batch_normalization_4/cond/batchnorm/add_1Add*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*/
_output_shapes
:€€€€€€€€€@@ *
T0
Ѕ
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*
N*1
_output_shapes
:€€€€€€€€€@@ : *
T0
u
up_sampling2d_2/ShapeShape batch_normalization_4/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_2/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape#up_sampling2d_2/strided_slice/stack%up_sampling2d_2/strided_slice/stack_1%up_sampling2d_2/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
f
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_2/mulMulup_sampling2d_2/strided_sliceup_sampling2d_2/Const*
_output_shapes
:*
T0
∆
%up_sampling2d_2/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_4/cond/Mergeup_sampling2d_2/mul*
align_corners( *
T0*1
_output_shapes
:€€€€€€€€€АА 
v
conv2d_5/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_5/random_uniform/minConst*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
≤
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: *
seed2ЩЃИ*
seed±€е)
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*&
_output_shapes
: *
T0
Й
conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*&
_output_shapes
: 
У
conv2d_5/kernel
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
»
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel
Ж
conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
: 
[
conv2d_5/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_5/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
≠
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:
t
conv2d_5/bias/readIdentityconv2d_5/bias*
T0* 
_class
loc:@conv2d_5/bias*
_output_shapes
:
s
"conv2d_5/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Е
conv2d_5/convolutionConv2D%up_sampling2d_2/ResizeNearestNeighborconv2d_5/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА
Ш
conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА*
T0
c
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*1
_output_shapes
:€€€€€€€€€АА
h
batch_normalization_5/ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
З
batch_normalization_5/gamma
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
д
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gammabatch_normalization_5/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:
Ю
 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
j
batch_normalization_5/Const_1Const*
valueB*    *
dtype0*
_output_shapes
:
Ж
batch_normalization_5/beta
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
г
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/betabatch_normalization_5/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes
:
Ы
batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:
j
batch_normalization_5/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:
Н
!batch_normalization_5/moving_mean
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ш
(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_meanbatch_normalization_5/Const_2*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(
∞
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
j
batch_normalization_5/Const_3Const*
valueB*  А?*
dtype0*
_output_shapes
:
С
%batch_normalization_5/moving_variance
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Д
,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variancebatch_normalization_5/Const_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
Љ
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
Й
4batch_normalization_5/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_5/moments/meanMeanconv2d_5/Relu4batch_normalization_5/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
П
*batch_normalization_5/moments/StopGradientStopGradient"batch_normalization_5/moments/mean*
T0*&
_output_shapes
:
ї
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferenceconv2d_5/Relu*batch_normalization_5/moments/StopGradient*
T0*1
_output_shapes
:€€€€€€€€€АА
Н
8batch_normalization_5/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_5/moments/varianceMean/batch_normalization_5/moments/SquaredDifference8batch_normalization_5/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
Т
%batch_normalization_5/moments/SqueezeSqueeze"batch_normalization_5/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:
Ш
'batch_normalization_5/moments/Squeeze_1Squeeze&batch_normalization_5/moments/variance*
T0*
_output_shapes
:*
squeeze_dims
 
j
%batch_normalization_5/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_5/batchnorm/addAdd'batch_normalization_5/moments/Squeeze_1%batch_normalization_5/batchnorm/add/y*
_output_shapes
:*
T0
x
%batch_normalization_5/batchnorm/RsqrtRsqrt#batch_normalization_5/batchnorm/add*
T0*
_output_shapes
:
Ш
#batch_normalization_5/batchnorm/mulMul%batch_normalization_5/batchnorm/Rsqrt batch_normalization_5/gamma/read*
T0*
_output_shapes
:
Ь
%batch_normalization_5/batchnorm/mul_1Mulconv2d_5/Relu#batch_normalization_5/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
Э
%batch_normalization_5/batchnorm/mul_2Mul%batch_normalization_5/moments/Squeeze#batch_normalization_5/batchnorm/mul*
T0*
_output_shapes
:
Ч
#batch_normalization_5/batchnorm/subSubbatch_normalization_5/beta/read%batch_normalization_5/batchnorm/mul_2*
T0*
_output_shapes
:
і
%batch_normalization_5/batchnorm/add_1Add%batch_normalization_5/batchnorm/mul_1#batch_normalization_5/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
¶
+batch_normalization_5/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_5/moving_mean
Џ
)batch_normalization_5/AssignMovingAvg/subSub&batch_normalization_5/moving_mean/read%batch_normalization_5/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
г
)batch_normalization_5/AssignMovingAvg/mulMul)batch_normalization_5/AssignMovingAvg/sub+batch_normalization_5/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
о
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_5/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
ђ
-batch_normalization_5/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_5/moving_variance
ж
+batch_normalization_5/AssignMovingAvg_1/subSub*batch_normalization_5/moving_variance/read'batch_normalization_5/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:
н
+batch_normalization_5/AssignMovingAvg_1/mulMul+batch_normalization_5/AssignMovingAvg_1/sub-batch_normalization_5/AssignMovingAvg_1/decay*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
ъ
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_5/AssignMovingAvg_1/mul*
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
™
!batch_normalization_5/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_5/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
Л
#batch_normalization_5/cond/Switch_1Switch%batch_normalization_5/batchnorm/add_1"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Х
*batch_normalization_5/cond/batchnorm/add/yConst$^batch_normalization_5/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *oГ:
±
(batch_normalization_5/cond/batchnorm/addAdd/batch_normalization_5/cond/batchnorm/add/Switch*batch_normalization_5/cond/batchnorm/add/y*
_output_shapes
:*
T0
о
/batch_normalization_5/cond/batchnorm/add/SwitchSwitch*batch_normalization_5/moving_variance/read"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance* 
_output_shapes
::
В
*batch_normalization_5/cond/batchnorm/RsqrtRsqrt(batch_normalization_5/cond/batchnorm/add*
T0*
_output_shapes
:
±
(batch_normalization_5/cond/batchnorm/mulMul*batch_normalization_5/cond/batchnorm/Rsqrt/batch_normalization_5/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:
Џ
/batch_normalization_5/cond/batchnorm/mul/SwitchSwitch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma* 
_output_shapes
::
 
*batch_normalization_5/cond/batchnorm/mul_1Mul1batch_normalization_5/cond/batchnorm/mul_1/Switch(batch_normalization_5/cond/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
й
1batch_normalization_5/cond/batchnorm/mul_1/SwitchSwitchconv2d_5/Relu"batch_normalization_5/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0* 
_class
loc:@conv2d_5/Relu
≥
*batch_normalization_5/cond/batchnorm/mul_2Mul1batch_normalization_5/cond/batchnorm/mul_2/Switch(batch_normalization_5/cond/batchnorm/mul*
T0*
_output_shapes
:
и
1batch_normalization_5/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_5/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean* 
_output_shapes
::
±
(batch_normalization_5/cond/batchnorm/subSub/batch_normalization_5/cond/batchnorm/sub/Switch*batch_normalization_5/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Ў
/batch_normalization_5/cond/batchnorm/sub/SwitchSwitchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_5/beta* 
_output_shapes
::
√
*batch_normalization_5/cond/batchnorm/add_1Add*batch_normalization_5/cond/batchnorm/mul_1(batch_normalization_5/cond/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
√
 batch_normalization_5/cond/MergeMerge*batch_normalization_5/cond/batchnorm/add_1%batch_normalization_5/cond/Switch_1:1*
T0*
N*3
_output_shapes!
:€€€€€€€€€АА: 
u
up_sampling2d_3/ShapeShape batch_normalization_5/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape#up_sampling2d_3/strided_slice/stack%up_sampling2d_3/strided_slice/stack_1%up_sampling2d_3/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
f
up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_3/mulMulup_sampling2d_3/strided_sliceup_sampling2d_3/Const*
T0*
_output_shapes
:
∆
%up_sampling2d_3/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_5/cond/Mergeup_sampling2d_3/mul*
align_corners( *
T0*1
_output_shapes
:€€€€€€€€€АА
v
conv2d_6/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
`
conv2d_6/random_uniform/minConst*
valueB
 *:ЌЊ*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Ќ>
≤
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2Ы€Ю*
seed±€е)
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:
Й
conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*
T0*&
_output_shapes
:
У
conv2d_6/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
»
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel
Ж
conv2d_6/kernel/readIdentityconv2d_6/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
[
conv2d_6/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_6/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Е
conv2d_6/convolutionConv2D%up_sampling2d_3/ResizeNearestNeighborconv2d_6/kernel/read*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ш
conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА
c
conv2d_6/TanhTanhconv2d_6/BiasAdd*
T0*1
_output_shapes
:€€€€€€€€€АА
]
RMSprop/lr/initial_valueConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
n

RMSprop/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
™
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
use_locking(*
T0*
_class
loc:@RMSprop/lr*
validate_shape(*
_output_shapes
: 
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
RMSprop/rho
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ѓ
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
use_locking(*
T0*
_class
loc:@RMSprop/rho*
validate_shape(*
_output_shapes
: 
j
RMSprop/rho/readIdentityRMSprop/rho*
T0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
`
RMSprop/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
RMSprop/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ґ
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
use_locking(*
T0* 
_class
loc:@RMSprop/decay*
validate_shape(*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
 
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
use_locking(*
T0	*%
_class
loc:@RMSprop/iterations*
validate_shape(*
_output_shapes
: 

RMSprop/iterations/readIdentityRMSprop/iterations*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
Є
conv2d_6_targetPlaceholder*?
shape6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
r
conv2d_6_sample_weightsPlaceholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
]
loss/conv2d_6_loss/ConstConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
]
loss/conv2d_6_loss/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
loss/conv2d_6_loss/subSubloss/conv2d_6_loss/sub/xloss/conv2d_6_loss/Const*
_output_shapes
: *
T0
Ц
(loss/conv2d_6_loss/clip_by_value/MinimumMinimumconv2d_6/Tanhloss/conv2d_6_loss/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
Ђ
 loss/conv2d_6_loss/clip_by_valueMaximum(loss/conv2d_6_loss/clip_by_value/Minimumloss/conv2d_6_loss/Const*
T0*1
_output_shapes
:€€€€€€€€€АА
_
loss/conv2d_6_loss/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
loss/conv2d_6_loss/sub_1Subloss/conv2d_6_loss/sub_1/x loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА*
T0
Э
loss/conv2d_6_loss/truedivRealDiv loss/conv2d_6_loss/clip_by_valueloss/conv2d_6_loss/sub_1*
T0*1
_output_shapes
:€€€€€€€€€АА
u
loss/conv2d_6_loss/LogLogloss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА*
T0
М
+loss/conv2d_6_loss/logistic_loss/zeros_like	ZerosLikeloss/conv2d_6_loss/Log*
T0*1
_output_shapes
:€€€€€€€€€АА
Њ
-loss/conv2d_6_loss/logistic_loss/GreaterEqualGreaterEqualloss/conv2d_6_loss/Log+loss/conv2d_6_loss/logistic_loss/zeros_like*
T0*1
_output_shapes
:€€€€€€€€€АА
б
'loss/conv2d_6_loss/logistic_loss/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualloss/conv2d_6_loss/Log+loss/conv2d_6_loss/logistic_loss/zeros_like*1
_output_shapes
:€€€€€€€€€АА*
T0

$loss/conv2d_6_loss/logistic_loss/NegNegloss/conv2d_6_loss/Log*
T0*1
_output_shapes
:€€€€€€€€€АА
№
)loss/conv2d_6_loss/logistic_loss/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqual$loss/conv2d_6_loss/logistic_loss/Negloss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА*
T0
Р
$loss/conv2d_6_loss/logistic_loss/mulMulloss/conv2d_6_loss/Logconv2d_6_target*
T0*1
_output_shapes
:€€€€€€€€€АА
ґ
$loss/conv2d_6_loss/logistic_loss/subSub'loss/conv2d_6_loss/logistic_loss/Select$loss/conv2d_6_loss/logistic_loss/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
Т
$loss/conv2d_6_loss/logistic_loss/ExpExp)loss/conv2d_6_loss/logistic_loss/Select_1*
T0*1
_output_shapes
:€€€€€€€€€АА
С
&loss/conv2d_6_loss/logistic_loss/Log1pLog1p$loss/conv2d_6_loss/logistic_loss/Exp*
T0*1
_output_shapes
:€€€€€€€€€АА
±
 loss/conv2d_6_loss/logistic_lossAdd$loss/conv2d_6_loss/logistic_loss/sub&loss/conv2d_6_loss/logistic_loss/Log1p*
T0*1
_output_shapes
:€€€€€€€€€АА
t
)loss/conv2d_6_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ѕ
loss/conv2d_6_loss/MeanMean loss/conv2d_6_loss/logistic_loss)loss/conv2d_6_loss/Mean/reduction_indices*
T0*-
_output_shapes
:€€€€€€€€€АА*

Tidx0*
	keep_dims( 
|
+loss/conv2d_6_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
≤
loss/conv2d_6_loss/Mean_1Meanloss/conv2d_6_loss/Mean+loss/conv2d_6_loss/Mean_1/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0

loss/conv2d_6_loss/mulMulloss/conv2d_6_loss/Mean_1conv2d_6_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
b
loss/conv2d_6_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
loss/conv2d_6_loss/NotEqualNotEqualconv2d_6_sample_weightsloss/conv2d_6_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
y
loss/conv2d_6_loss/CastCastloss/conv2d_6_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
d
loss/conv2d_6_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
loss/conv2d_6_loss/Mean_2Meanloss/conv2d_6_loss/Castloss/conv2d_6_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
И
loss/conv2d_6_loss/truediv_1RealDivloss/conv2d_6_loss/mulloss/conv2d_6_loss/Mean_2*#
_output_shapes
:€€€€€€€€€*
T0
d
loss/conv2d_6_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Щ
loss/conv2d_6_loss/Mean_3Meanloss/conv2d_6_loss/truediv_1loss/conv2d_6_loss/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/conv2d_6_loss/Mean_3*
T0*
_output_shapes
: 
А
 training/RMSprop/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
Ж
$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  А?*
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
њ
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
≠
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/conv2d_6_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
†
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
њ
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
dtype0*
_output_shapes
:
¶
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
…
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ShapeShapeloss/conv2d_6_loss/truediv_1*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
Є
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/TileTileAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
Ћ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_6_loss/truediv_1*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
≤
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
Ј
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
dtype0*
_output_shapes
:
ґ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ProdProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_1?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
dtype0*
_output_shapes
:
Ї
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Prod_1ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_2Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const_1*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
≥
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
Ґ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/MaximumMaximum@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Prod_1Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: 
†
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/floordivFloorDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: 
и
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/CastCastBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/floordiv*

SrcT0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: *

DstT0
®
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivRealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Tile>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
…
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ShapeShapeloss/conv2d_6_loss/mul*
_output_shapes
:*
T0*
out_type0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
Є
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1Const*
valueB */
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*
dtype0*
_output_shapes
: 
г
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ShapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDivRealDivAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivloss/conv2d_6_loss/Mean_2*#
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
“
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/SumSumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDivRtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
¬
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ReshapeReshape@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/SumBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*
Tshape0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
Њ
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/NegNegloss/conv2d_6_loss/mul*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Н
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_1RealDiv@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Negloss/conv2d_6_loss/Mean_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
У
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_2RealDivFtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_1loss/conv2d_6_loss/Mean_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
±
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/mulMulAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivFtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
“
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Sum_1Sum@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/mulTtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
ї
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshape_1ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Sum_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
ј
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ShapeShapeloss/conv2d_6_loss/Mean_1*
T0*
out_type0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:
ј
>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1Shapeconv2d_6_sample_weights*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/conv2d_6_loss/mul
Ћ
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
щ
:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/MulMulDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshapeconv2d_6_sample_weights*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
ґ
:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/SumSum:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/MulLtraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
™
>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ReshapeReshape:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape*
T0*
Tshape0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
э
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Mul_1Mulloss/conv2d_6_loss/Mean_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshape*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
Љ
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum_1Sum<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Mul_1Ntraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
∞
@training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Reshape_1Reshape<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum_1>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
ƒ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ShapeShapeloss/conv2d_6_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
Ѓ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/SizeConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
Д
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/addAdd+loss/conv2d_6_loss/Mean_1/reduction_indices>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
Ы
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/modFloorMod=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/add>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_1Const*
valueB:*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
:
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/deltaConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
м
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/rangeRangeEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/start>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/SizeEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/delta*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:*

Tidx0
і
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill/valueConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
і
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/FillFillAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill/value*
T0*

index_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
Њ
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitchDynamicStitch?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/mod?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
N*#
_output_shapes
:€€€€€€€€€
≥
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
ґ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/MaximumMaximumGtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitchCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
•
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordivFloorDiv?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ShapeAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
і
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ReshapeReshape>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ReshapeGtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
’
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/TileTileAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
∆
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_6_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
»
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_6_loss/Mean_1*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
Ј
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
ґ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ProdProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_2?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
Ї
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Prod_1ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_3Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
¶
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1Maximum@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Prod_1Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1/y*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
§
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv_1FloorDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ProdCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
к
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/CastCastDtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv_1*

SrcT0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: *

DstT0
≤
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/truedivRealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Tile>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*-
_output_shapes
:€€€€€€€€€АА
…
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ShapeShape loss/conv2d_6_loss/logistic_loss*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean
™
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/SizeConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
ш
;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/addAdd)loss/conv2d_6_loss/Mean/reduction_indices<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
П
;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/modFloorMod;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/add<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_1Const*
valueB **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/startConst*
value	B : **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean
в
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/rangeRangeCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/start<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/SizeCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0**
_class 
loc:@loss/conv2d_6_loss/Mean
∞
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean
®
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/FillFill?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_1Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*

index_type0**
_class 
loc:@loss/conv2d_6_loss/Mean
≤
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitchDynamicStitch=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/mod=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
N*#
_output_shapes
:€€€€€€€€€
ѓ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum/yConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/MaximumMaximumEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitchAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum/y*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*#
_output_shapes
:€€€€€€€€€
Э
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordivFloorDiv=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
±
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ReshapeReshapeAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/truedivEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
Џ
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/TileTile?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Reshape@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
Ћ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_2Shape loss/conv2d_6_loss/logistic_loss*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean
¬
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_3Shapeloss/conv2d_6_loss/Mean*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
≥
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ConstConst*
valueB: **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
:
Ѓ
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ProdProd?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_2=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
µ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const_1Const*
valueB: **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
:
≤
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Prod_1Prod?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_3?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1/yConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
Ю
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1Maximum>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Prod_1Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1/y*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
Ь
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv_1FloorDiv<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0**
_class 
loc:@loss/conv2d_6_loss/Mean
д
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/CastCastBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0**
_class 
loc:@loss/conv2d_6_loss/Mean
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivRealDiv<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Tile<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Cast*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*1
_output_shapes
:€€€€€€€€€АА
я
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ShapeShape$loss/conv2d_6_loss/logistic_loss/sub*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*
_output_shapes
:
г
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1Shape&loss/conv2d_6_loss/logistic_loss/Log1p*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss
у
Vtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ў
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/SumSum?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivVtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*
_output_shapes
:*

Tidx0*
	keep_dims( 
а
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ReshapeReshapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/SumFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*1
_output_shapes
:€€€€€€€€€АА
Ё
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Sum_1Sum?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivXtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss
ж
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1ReshapeFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Sum_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss
к
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ShapeShape'loss/conv2d_6_loss/logistic_loss/Select*
_output_shapes
:*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub
й
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1Shape$loss/conv2d_6_loss/logistic_loss/mul*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
Г
Ztraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ShapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/SumSumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ReshapeZtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
р
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ReshapeReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub
т
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Sum_1SumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape\training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/NegNegJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Sum_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
ф
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1ReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/NegLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*1
_output_shapes
:€€€€€€€€€АА
Щ
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add/xConstK^training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1*
dtype0*
_output_shapes
: *
valueB
 *  А?*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p
Љ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/addAddLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add/x$loss/conv2d_6_loss/logistic_loss/Exp*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p*1
_output_shapes
:€€€€€€€€€АА
Ґ
Qtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/Reciprocal
ReciprocalJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add*1
_output_shapes
:€€€€€€€€€АА*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p
з
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/mulMulJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1Qtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/Reciprocal*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p*1
_output_shapes
:€€€€€€€€€АА
п
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_like	ZerosLikeloss/conv2d_6_loss/Log*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*1
_output_shapes
:€€€€€€€€€АА
°
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ReshapeRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_like*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*1
_output_shapes
:€€€€€€€€€АА
£
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqualRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_likeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape*1
_output_shapes
:€€€€€€€€€АА*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select
ў
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ShapeShapeloss/conv2d_6_loss/Log*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*
_output_shapes
:
‘
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1Shapeconv2d_6_target*
_output_shapes
:*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
Г
Ztraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ShapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
•
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/MulMulNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1conv2d_6_target*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА
о
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/SumSumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/MulZtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
р
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ReshapeReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА
Ѓ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Mul_1Mulloss/conv2d_6_loss/LogNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА
ф
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Sum_1SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Mul_1\training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
П
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Reshape_1ReshapeJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Sum_1Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ґ
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mulMulJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/mul$loss/conv2d_6_loss/logistic_loss/Exp*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/Exp*1
_output_shapes
:€€€€€€€€€АА
Б
Ttraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_like	ZerosLike$loss/conv2d_6_loss/logistic_loss/Neg*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1*1
_output_shapes
:€€€€€€€€€АА
£
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mulTtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_like*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1*1
_output_shapes
:€€€€€€€€€АА
•
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqualTtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_likeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mul*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1*1
_output_shapes
:€€€€€€€€€АА
Ц
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Neg_grad/NegNegPtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/Neg*1
_output_shapes
:€€€€€€€€€АА
д
training/RMSprop/gradients/AddNAddNNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/SelectLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ReshapeRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Neg_grad/Neg*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*
N*1
_output_shapes
:€€€€€€€€€АА
ф
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/Reciprocal
Reciprocalloss/conv2d_6_loss/truediv ^training/RMSprop/gradients/AddN*
T0*)
_class
loc:@loss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА
М
:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulMultraining/RMSprop/gradients/AddNAtraining/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/Reciprocal*
T0*)
_class
loc:@loss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА
ѕ
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ShapeShape loss/conv2d_6_loss/clip_by_value*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
_output_shapes
:
…
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1Shapeloss/conv2d_6_loss/sub_1*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
_output_shapes
:
џ
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ShapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDivRealDiv:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulloss/conv2d_6_loss/sub_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
 
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/SumSumBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDivPtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgs*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
»
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ReshapeReshape>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
“
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/NegNeg loss/conv2d_6_loss/clip_by_value*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
Ф
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_1RealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Negloss/conv2d_6_loss/sub_1*1
_output_shapes
:€€€€€€€€€АА*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
Ъ
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_2RealDivDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_1loss/conv2d_6_loss/sub_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
≤
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/mulMul:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
 
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum_1Sum>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/mulRtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
_output_shapes
:
ќ
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Reshape@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum_1Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
Ѓ
>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB *+
_class!
loc:@loss/conv2d_6_loss/sub_1
Ќ
@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1Shape loss/conv2d_6_loss/clip_by_value*
T0*
out_type0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:
”
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∆
<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/SumSumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Ntraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgs*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
•
@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/ReshapeReshape<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0*+
_class!
loc:@loss/conv2d_6_loss/sub_1
 
>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum_1SumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Ptraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
”
<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/NegNeg>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum_1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:
ƒ
Btraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Reshape_1Reshape<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Neg@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*1
_output_shapes
:€€€€€€€€€АА
•
!training/RMSprop/gradients/AddN_1AddNBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Reshape_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
N*1
_output_shapes
:€€€€€€€€€АА
г
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ShapeShape(loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
ј
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1Const*
valueB *3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
dtype0*
_output_shapes
: 
ё
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_2Shape!training/RMSprop/gradients/AddN_1*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
:
∆
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
dtype0*
_output_shapes
: 
й
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zerosFillHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_2Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros/Const*1
_output_shapes
:€€€€€€€€€АА*
T0*

index_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
Т
Mtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqualGreaterEqual(loss/conv2d_6_loss/clip_by_value/Minimumloss/conv2d_6_loss/Const*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
у
Vtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ь
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SelectSelectMtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqual!training/RMSprop/gradients/AddN_1Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
ю
Itraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Select_1SelectMtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqualFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros!training/RMSprop/gradients/AddN_1*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
б
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SumSumGtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SelectVtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
:*

Tidx0*
	keep_dims( 
а
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ReshapeReshapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SumFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
з
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Sum_1SumItraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Select_1Xtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape_1ReshapeFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Sum_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
Ў
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ShapeShapeconv2d_6/Tanh*
T0*
out_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:
–
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 
Х
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_2ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape*
T0*
out_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:
÷
Ttraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 
Й
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zerosFillPtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_2Ttraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
€
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualconv2d_6/Tanhloss/conv2d_6_loss/sub*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
У
^training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ShapePtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
ј
Otraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SelectSelectRtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqualHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ReshapeNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
¬
Qtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select_1SelectRtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqualNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zerosHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
Б
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SumSumOtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select^training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
А
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ReshapeReshapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SumNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
З
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Sum_1SumQtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select_1`training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:*

Tidx0*
	keep_dims( 
л
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Sum_1Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
: 
Б
6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGradTanhGradconv2d_6/TanhPtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Reshape*
T0* 
_class
loc:@conv2d_6/Tanh*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:
п
;training/RMSprop/gradients/conv2d_6/convolution_grad/ShapeNShapeN%up_sampling2d_3/ResizeNearestNeighborconv2d_6/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_6/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_6/convolution_grad/ConstConst*%
valueB"            *'
_class
loc:@conv2d_6/convolution*
dtype0*
_output_shapes
:
љ
Htraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ƒ
Itraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_3/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_6/convolution_grad/Const6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
п
dtraining/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"А   А   *8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ј
_training/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*1
_output_shapes
:€€€€€€€€€АА
м
Jtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_5/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor
щ
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_5/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1
В
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
И
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:
Й
!training/RMSprop/gradients/SwitchSwitch%batch_normalization_5/batchnorm/add_1"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
љ
"training/RMSprop/gradients/Shape_1Shape!training/RMSprop/gradients/Switch*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:
•
&training/RMSprop/gradients/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
ь
 training/RMSprop/gradients/zerosFill"training/RMSprop/gradients/Shape_1&training/RMSprop/gradients/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
«
Mtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_gradMerge training/RMSprop/gradients/zerosLtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
А
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_5/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape(batch_normalization_5/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1
И
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ў
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_5/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
М
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1
ч
Ttraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:
Л
Ltraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/sub*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ShapeShape%batch_normalization_5/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
ц
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
ф
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
ъ
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:
џ
#training/RMSprop/gradients/Switch_1Switchconv2d_5/Relu"batch_normalization_5/cond/pred_id*
T0* 
_class
loc:@conv2d_5/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
©
"training/RMSprop/gradients/Shape_2Shape%training/RMSprop/gradients/Switch_1:1*
T0*
out_type0* 
_class
loc:@conv2d_5/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_1/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_5/Relu*
dtype0*
_output_shapes
: 
и
"training/RMSprop/gradients/zeros_1Fill"training/RMSprop/gradients/Shape_2(training/RMSprop/gradients/zeros_1/Const*
T0*

index_type0* 
_class
loc:@conv2d_5/Relu*1
_output_shapes
:€€€€€€€€€АА
≈
[training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_1*
T0* 
_class
loc:@conv2d_5/Relu*
N*3
_output_shapes!
:€€€€€€€€€АА: 
ћ
#training/RMSprop/gradients/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_5/beta* 
_output_shapes
::
ґ
"training/RMSprop/gradients/Shape_3Shape%training/RMSprop/gradients/Switch_2:1*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@batch_normalization_5/beta
Ь
(training/RMSprop/gradients/zeros_2/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
: 
ё
"training/RMSprop/gradients/zeros_2Fill"training/RMSprop/gradients/Shape_3(training/RMSprop/gradients/zeros_2/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:
ї
Ytraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1"training/RMSprop/gradients/zeros_2*
T0*-
_class#
!loc:@batch_normalization_5/beta*
N*
_output_shapes

:: 
±
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/Neg(batch_normalization_5/cond/batchnorm/mul*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_2
Љ
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/Neg1batch_normalization_5/cond/batchnorm/mul_2/Switch*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_2
“
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ShapeShapeconv2d_5/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1Const*
valueB:*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ї
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape#batch_normalization_5/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
т
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
¶
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul_1Mulconv2d_5/ReluMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ш
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
ь
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/sub*
_output_shapes
:
Њ
!training/RMSprop/gradients/AddN_2AddNTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
N*
_output_shapes
:
Й
Ltraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_2/batch_normalization_5/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/mul*
_output_shapes
:
Ж
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_2*batch_normalization_5/cond/batchnorm/Rsqrt*
_output_shapes
:*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/mul
≤
!training/RMSprop/gradients/AddN_3AddNYtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_5/beta*
N*
_output_shapes
:
Э
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/Neg#batch_normalization_5/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_2*
_output_shapes
:
°
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/Neg%batch_normalization_5/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_2*
_output_shapes
:
ќ
#training/RMSprop/gradients/Switch_3Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma* 
_output_shapes
::
Ј
"training/RMSprop/gradients/Shape_4Shape%training/RMSprop/gradients/Switch_3:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
Э
(training/RMSprop/gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma
я
"training/RMSprop/gradients/zeros_3Fill"training/RMSprop/gradients/Shape_4(training/RMSprop/gradients/zeros_3/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
ґ
Ytraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/Mul_1"training/RMSprop/gradients/zeros_3*
N*
_output_shapes

:: *
T0*.
_class$
" loc:@batch_normalization_5/gamma
ё
Ktraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            *8
_class.
,*loc:@batch_normalization_5/moments/Squeeze
й
Mtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/moments/Squeeze*&
_output_shapes
:
ѓ
!training/RMSprop/gradients/AddN_4AddNOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
р
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_4 batch_normalization_5/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/mul*
_output_shapes
:
ч
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_4%batch_normalization_5/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/mul*
_output_shapes
:
Ђ
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_5/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/Rsqrt*
_output_shapes
:
≠
!training/RMSprop/gradients/AddN_5AddNYtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul_1*
N*
_output_shapes
:*
T0*.
_class$
" loc:@batch_normalization_5/gamma
Ћ
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ShapeConst*
valueB:*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add
’
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
_output_shapes
:
ц
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
„
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ShapeConst*%
valueB"            *:
_class0
.,loc:@batch_normalization_5/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/Shape*&
_output_shapes
:*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_5/moments/Squeeze_1
ц
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeShape/batch_normalization_5/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/addAdd8batch_normalization_5/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *9
_class/
-+loc:@batch_normalization_5/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_5/moments/variance
ќ
Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance
и
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
Ќ
Ptraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
ш
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_2Shape/batch_normalization_5/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_3Const*%
valueB"            *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_5/moments/variance
о
Mtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: *

DstT0
к
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*1
_output_shapes
:€€€€€€€€€АА
ж
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ShapeShapeconv2d_5/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
_output_shapes
:
ф
Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1Const*%
valueB"            *B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
dtype0*
_output_shapes
: 
В
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
ж
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/subSubconv2d_5/Relu*batch_normalization_5/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ж
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ь
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ь
Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
†
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*&
_output_shapes
:
™
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*&
_output_shapes
:
ћ
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeShapeconv2d_5/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/addAdd4batch_normalization_5/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean
Щ
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean
ф
Ptraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean
…
Ktraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_5/moments/mean
Ж
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
ќ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_2Shapeconv2d_5/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_3Const*%
valueB"            *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_5/moments/mean
ё
Itraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean
 
Ltraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: *

DstT0
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*1
_output_shapes
:€€€€€€€€€АА
б
!training/RMSprop/gradients/AddN_6AddN[training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_5/Relu*
N*1
_output_shapes
:€€€€€€€€€АА
“
6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGradReluGrad!training/RMSprop/gradients/AddN_6conv2d_5/Relu*
T0* 
_class
loc:@conv2d_5/Relu*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_5/BiasAdd
п
;training/RMSprop/gradients/conv2d_5/convolution_grad/ShapeNShapeN%up_sampling2d_2/ResizeNearestNeighborconv2d_5/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_5/convolution
Љ
:training/RMSprop/gradients/conv2d_5/convolution_grad/ConstConst*%
valueB"             *'
_class
loc:@conv2d_5/convolution*
dtype0*
_output_shapes
:
љ
Htraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/read6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА *
	dilations

ƒ
Itraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_2/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_5/convolution_grad/Const6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
п
dtraining/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"@   @   *8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
µ
_training/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*/
_output_shapes
:€€€€€€€€€@@ 
и
Jtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
щ
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_4/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:
Ж
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
Ж
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
_output_shapes
: 
З
#training/RMSprop/gradients/Switch_4Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
њ
"training/RMSprop/gradients/Shape_5Shape#training/RMSprop/gradients/Switch_4*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
І
(training/RMSprop/gradients/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
ю
"training/RMSprop/gradients/zeros_4Fill"training/RMSprop/gradients/Shape_5(training/RMSprop/gradients/zeros_4/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
«
Mtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradMerge"training/RMSprop/gradients/zeros_4Ltraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
N*1
_output_shapes
:€€€€€€€€€@@ : 
А
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_4/cond/batchnorm/mul_1/Switch*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
џ
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
Ы
`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
Ж
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
„
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape*/
_output_shapes
:€€€€€€€€€@@ *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
М
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
ч
Ttraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
: 
Л
Ltraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/sub*
_output_shapes
: 
к
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Const*
valueB: *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
ц
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
т
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
ъ
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
: 
„
#training/RMSprop/gradients/Switch_5Switchconv2d_4/Relu"batch_normalization_4/cond/pred_id*
T0* 
_class
loc:@conv2d_4/Relu*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
©
"training/RMSprop/gradients/Shape_6Shape%training/RMSprop/gradients/Switch_5:1*
T0*
out_type0* 
_class
loc:@conv2d_4/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_5/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2d_4/Relu
ж
"training/RMSprop/gradients/zeros_5Fill"training/RMSprop/gradients/Shape_6(training/RMSprop/gradients/zeros_5/Const*
T0*

index_type0* 
_class
loc:@conv2d_4/Relu*/
_output_shapes
:€€€€€€€€€@@ 
√
[training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_5*
N*1
_output_shapes
:€€€€€€€€€@@ : *
T0* 
_class
loc:@conv2d_4/Relu
ћ
#training/RMSprop/gradients/Switch_6Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
: : 
ґ
"training/RMSprop/gradients/Shape_7Shape%training/RMSprop/gradients/Switch_6:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:
Ь
(training/RMSprop/gradients/zeros_6/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
ё
"training/RMSprop/gradients/zeros_6Fill"training/RMSprop/gradients/Shape_7(training/RMSprop/gradients/zeros_6/Const*
_output_shapes
: *
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta
ї
Ytraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1"training/RMSprop/gradients/zeros_6*
N*
_output_shapes

: : *
T0*-
_class#
!loc:@batch_normalization_4/beta
±
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg(batch_normalization_4/cond/batchnorm/mul*
_output_shapes
: *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2
Љ
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg1batch_normalization_4/cond/batchnorm/mul_2/Switch*
_output_shapes
: *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2
“
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapeconv2d_4/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Const*
valueB: *8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Є
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape#batch_normalization_4/batchnorm/mul*/
_output_shapes
:€€€€€€€€€@@ *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
т
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
§
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Mulconv2d_4/ReluMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
ш
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:
г
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
: 
ь
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/sub*
_output_shapes
: 
Њ
!training/RMSprop/gradients/AddN_7AddNTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
N*
_output_shapes
: 
Й
Ltraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_7/batch_normalization_4/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes
: 
Ж
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_7*batch_normalization_4/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes
: 
≤
!training/RMSprop/gradients/AddN_8AddNYtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*
_output_shapes
: 
Э
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/Neg#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes
: 
°
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/Neg%batch_normalization_4/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes
: 
ќ
#training/RMSprop/gradients/Switch_7Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
: : 
Ј
"training/RMSprop/gradients/Shape_8Shape%training/RMSprop/gradients/Switch_7:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:
Э
(training/RMSprop/gradients/zeros_7/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma
я
"training/RMSprop/gradients/zeros_7Fill"training/RMSprop/gradients/Shape_8(training/RMSprop/gradients/zeros_7/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: 
ґ
Ytraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1"training/RMSprop/gradients/zeros_7*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
N*
_output_shapes

: : 
ё
Ktraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst*%
valueB"             *8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*&
_output_shapes
: 
ѓ
!training/RMSprop/gradients/AddN_9AddNOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
N*
_output_shapes
: 
р
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_9 batch_normalization_4/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes
: 
ч
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_9%batch_normalization_4/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes
: 
Ђ
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/Rsqrt*
_output_shapes
: 
Ѓ
"training/RMSprop/gradients/AddN_10AddNYtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1*
N*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
Ћ
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ShapeConst*
valueB: *6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *6
_class,
*(loc:@batch_normalization_4/batchnorm/add
€
Ytraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
’
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: 
ц
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add
„
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst*%
valueB"             *:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*&
_output_shapes
: 
ц
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance
Є
Jtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/addAdd8batch_normalization_4/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:*

Tidx0
ќ
Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_4/moments/variance
Ц
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_3Const*%
valueB"             *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
”
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance
о
Mtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance
Џ
Ptraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
Ў
Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

DstT0
и
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Cast*/
_output_shapes
:€€€€€€€€€@@ *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
ж
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapeconv2d_4/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Const*%
valueB"             *B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
dtype0*
_output_shapes
: 
А
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*/
_output_shapes
:€€€€€€€€€@@ *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
д
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/subSubconv2d_4/Relu*batch_normalization_4/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Д
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Ь
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ъ
Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
†
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
Ч
Ytraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*&
_output_shapes
: 
™
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*&
_output_shapes
: *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
ћ
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeShapeconv2d_4/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean
®
Ftraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/addAdd4batch_normalization_4/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean
Щ
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean
Ў
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
ё
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_4/moments/mean
Ж
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ќ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapeconv2d_4/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_3Const*%
valueB"             *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean
Џ
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
»
Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

DstT0
Ў
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*/
_output_shapes
:€€€€€€€€€@@ 
а
"training/RMSprop/gradients/AddN_11AddN[training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_4/Relu*
N*/
_output_shapes
:€€€€€€€€€@@ 
—
6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_11conv2d_4/Relu*
T0* 
_class
loc:@conv2d_4/Relu*/
_output_shapes
:€€€€€€€€€@@ 
д
<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_4/BiasAdd*
data_formatNHWC*
_output_shapes
: 
п
;training/RMSprop/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_4/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_4/convolution_grad/ConstConst*%
valueB"      @       *'
_class
loc:@conv2d_4/convolution*
dtype0*
_output_shapes
:
ї
Htraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/read6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€@@@*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ƒ
Itraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_4/convolution_grad/Const6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
п
dtraining/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*
valueB"        *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
µ
_training/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*/
_output_shapes
:€€€€€€€€€  @*
align_corners( 
и
Jtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
щ
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
џ
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:@*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
Ж
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
З
#training/RMSprop/gradients/Switch_8Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
њ
"training/RMSprop/gradients/Shape_9Shape#training/RMSprop/gradients/Switch_8*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
І
(training/RMSprop/gradients/zeros_8/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
dtype0*
_output_shapes
: 
ю
"training/RMSprop/gradients/zeros_8Fill"training/RMSprop/gradients/Shape_9(training/RMSprop/gradients/zeros_8/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
«
Mtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradMerge"training/RMSprop/gradients/zeros_8Ltraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1*
N*1
_output_shapes
:€€€€€€€€€  @: *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
А
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_3/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB:@*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape(batch_normalization_3/cond/batchnorm/mul*/
_output_shapes
:€€€€€€€€€  @*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
Ж
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
„
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape*/
_output_shapes
:€€€€€€€€€  @*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
М
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
Л
Ltraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/sub*
_output_shapes
:@
к
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeShape%batch_normalization_3/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Const*
valueB:@*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
т
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
ъ
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:@
„
#training/RMSprop/gradients/Switch_9Switchconv2d_3/Relu"batch_normalization_3/cond/pred_id*
T0* 
_class
loc:@conv2d_3/Relu*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
™
#training/RMSprop/gradients/Shape_10Shape%training/RMSprop/gradients/Switch_9:1*
T0*
out_type0* 
_class
loc:@conv2d_3/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_9/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_3/Relu*
dtype0*
_output_shapes
: 
з
"training/RMSprop/gradients/zeros_9Fill#training/RMSprop/gradients/Shape_10(training/RMSprop/gradients/zeros_9/Const*
T0*

index_type0* 
_class
loc:@conv2d_3/Relu*/
_output_shapes
:€€€€€€€€€  @
√
[training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_9*
N*1
_output_shapes
:€€€€€€€€€  @: *
T0* 
_class
loc:@conv2d_3/Relu
Ќ
$training/RMSprop/gradients/Switch_10Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
:@:@
Є
#training/RMSprop/gradients/Shape_11Shape&training/RMSprop/gradients/Switch_10:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_10/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
б
#training/RMSprop/gradients/zeros_10Fill#training/RMSprop/gradients/Shape_11)training/RMSprop/gradients/zeros_10/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:@
Љ
Ytraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_10*
N*
_output_shapes

:@: *
T0*-
_class#
!loc:@batch_normalization_3/beta
±
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:@
Љ
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg1batch_normalization_3/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:@
“
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapeconv2d_3/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1Const*
valueB:@*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
Є
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
т
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
§
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Mulconv2d_3/ReluMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
ш
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
ь
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/sub*
_output_shapes
:@
њ
"training/RMSprop/gradients/AddN_12AddNTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes
:@*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
К
Ltraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_12/batch_normalization_3/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:@
З
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_12*batch_normalization_3/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:@
≥
"training/RMSprop/gradients/AddN_13AddNYtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_3/beta*
N*
_output_shapes
:@
Э
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/Neg#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes
:@
°
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/Neg%batch_normalization_3/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes
:@
ѕ
$training/RMSprop/gradients/Switch_11Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:@:@
є
#training/RMSprop/gradients/Shape_12Shape&training/RMSprop/gradients/Switch_11:1*
_output_shapes
:*
T0*
out_type0*.
_class$
" loc:@batch_normalization_3/gamma
Ю
)training/RMSprop/gradients/zeros_11/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma
в
#training/RMSprop/gradients/zeros_11Fill#training/RMSprop/gradients/Shape_12)training/RMSprop/gradients/zeros_11/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@
Ј
Ytraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_11*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*
_output_shapes

:@: 
ё
Ktraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   *8
_class.
,*loc:@batch_normalization_3/moments/Squeeze
й
Mtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*&
_output_shapes
:@
∞
"training/RMSprop/gradients/AddN_14AddNOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
N*
_output_shapes
:@
с
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_14 batch_normalization_3/gamma/read*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul
ш
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_14%batch_normalization_3/batchnorm/Rsqrt*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul
Ђ
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_3/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/Rsqrt*
_output_shapes
:@
Ѓ
"training/RMSprop/gradients/AddN_15AddNYtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*
_output_shapes
:@
Ћ
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ShapeConst*
valueB:@*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgs*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
’
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape*
_output_shapes
:@*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add
ц
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
„
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ShapeConst*%
valueB"         @   *:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/Shape*&
_output_shapes
:@*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1
ц
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeShape/batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance
»
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/addAdd8batch_normalization_3/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_3/moments/variance
ќ
Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
ш
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_2Shape/batch_normalization_3/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_3Const*%
valueB"         @   *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance
о
Mtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance
Ў
Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: *

DstT0
и
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*/
_output_shapes
:€€€€€€€€€  @
ж
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeShapeconv2d_3/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1Const*%
valueB"         @   *B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
dtype0*
_output_shapes
: 
А
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*/
_output_shapes
:€€€€€€€€€  @*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
д
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubconv2d_3/Relu*batch_normalization_3/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
Д
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
Ь
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ъ
Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
†
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*&
_output_shapes
:@
™
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*&
_output_shapes
:@
ћ
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeShapeconv2d_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/addAdd4batch_normalization_3/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_3/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
Щ
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
Ў
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
Џ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
…
Ktraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
ќ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_2Shapeconv2d_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*%
valueB"         @   *5
_class+
)'loc:@batch_normalization_3/moments/mean
…
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean
ё
Itraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
»
Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
Е
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

DstT0
Ў
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Cast*/
_output_shapes
:€€€€€€€€€  @*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
а
"training/RMSprop/gradients/AddN_16AddN[training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_3/Relu*
N*/
_output_shapes
:€€€€€€€€€  @
—
6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_16conv2d_3/Relu*
T0* 
_class
loc:@conv2d_3/Relu*/
_output_shapes
:€€€€€€€€€  @
д
<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes
:@
к
;training/RMSprop/gradients/conv2d_3/convolution_grad/ShapeNShapeN batch_normalization_2/cond/Mergeconv2d_3/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution
Љ
:training/RMSprop/gradients/conv2d_3/convolution_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"          @   *'
_class
loc:@conv2d_3/convolution
ї
Htraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ 
њ
Itraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter batch_normalization_2/cond/Merge:training/RMSprop/gradients/conv2d_3/convolution_grad/Const6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
ј
Jtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchHtraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropInput"batch_normalization_2/cond/pred_id*
T0*'
_class
loc:@conv2d_3/convolution*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
щ
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
В
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
Ж
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
: 
И
$training/RMSprop/gradients/Switch_12Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Ѕ
#training/RMSprop/gradients/Shape_13Shape$training/RMSprop/gradients/Switch_12*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
®
)training/RMSprop/gradients/zeros_12/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
dtype0*
_output_shapes
: 
Б
#training/RMSprop/gradients/zeros_12Fill#training/RMSprop/gradients/Shape_13)training/RMSprop/gradients/zeros_12/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
»
Mtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradMerge#training/RMSprop/gradients/zeros_12Ltraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
N*1
_output_shapes
:€€€€€€€€€@@ : 
А
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_2/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
Ж
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
„
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
М
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
ч
Ttraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
Л
Ltraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/sub*
_output_shapes
: 
к
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeShape%batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
—
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1Const*
valueB: *8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
т
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
ъ
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
: 
Ў
$training/RMSprop/gradients/Switch_13Switchconv2d_2/Relu"batch_normalization_2/cond/pred_id*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ *
T0* 
_class
loc:@conv2d_2/Relu
Ђ
#training/RMSprop/gradients/Shape_14Shape&training/RMSprop/gradients/Switch_13:1*
T0*
out_type0* 
_class
loc:@conv2d_2/Relu*
_output_shapes
:
Р
)training/RMSprop/gradients/zeros_13/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_2/Relu*
dtype0*
_output_shapes
: 
й
#training/RMSprop/gradients/zeros_13Fill#training/RMSprop/gradients/Shape_14)training/RMSprop/gradients/zeros_13/Const*
T0*

index_type0* 
_class
loc:@conv2d_2/Relu*/
_output_shapes
:€€€€€€€€€@@ 
ƒ
[training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape#training/RMSprop/gradients/zeros_13*
T0* 
_class
loc:@conv2d_2/Relu*
N*1
_output_shapes
:€€€€€€€€€@@ : 
Ќ
$training/RMSprop/gradients/Switch_14Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
: : 
Є
#training/RMSprop/gradients/Shape_15Shape&training/RMSprop/gradients/Switch_14:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta
б
#training/RMSprop/gradients/zeros_14Fill#training/RMSprop/gradients/Shape_15)training/RMSprop/gradients/zeros_14/Const*
_output_shapes
: *
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta
Љ
Ytraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_14*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*
_output_shapes

: : 
±
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes
: 
Љ
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg1batch_normalization_2/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes
: 
“
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeShapeconv2d_2/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1Const*
valueB: *8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
Є
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape#batch_normalization_2/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
т
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
т
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
§
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Mulconv2d_2/ReluMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
ш
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
г
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
: 
ь
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/sub
њ
"training/RMSprop/gradients/AddN_17AddNTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
N*
_output_shapes
: 
К
Ltraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_17/batch_normalization_2/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul*
_output_shapes
: 
З
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_17*batch_normalization_2/cond/batchnorm/Rsqrt*
_output_shapes
: *
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul
≥
"training/RMSprop/gradients/AddN_18AddNYtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*
_output_shapes
: 
Э
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/Neg#batch_normalization_2/batchnorm/mul*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2
°
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/Neg%batch_normalization_2/moments/Squeeze*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2
ѕ
$training/RMSprop/gradients/Switch_15Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id* 
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_2/gamma
є
#training/RMSprop/gradients/Shape_16Shape&training/RMSprop/gradients/Switch_15:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
Ю
)training/RMSprop/gradients/zeros_15/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma
в
#training/RMSprop/gradients/zeros_15Fill#training/RMSprop/gradients/Shape_16)training/RMSprop/gradients/zeros_15/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
Ј
Ytraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_15*
N*
_output_shapes

: : *
T0*.
_class$
" loc:@batch_normalization_2/gamma
ё
Ktraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ShapeConst*%
valueB"             *8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*&
_output_shapes
: 
∞
"training/RMSprop/gradients/AddN_19AddNOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
с
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_19 batch_normalization_2/gamma/read*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul
ш
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_19%batch_normalization_2/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes
: 
Ђ
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_2/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/Rsqrt*
_output_shapes
: 
Ѓ
"training/RMSprop/gradients/AddN_20AddNYtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1*
N*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_2/gamma
Ћ
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB: *6
_class,
*(loc:@batch_normalization_2/batchnorm/add
∆
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add
’
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
ц
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add
„
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ShapeConst*%
valueB"             *:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/Shape*&
_output_shapes
: *
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1
ц
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeShape/batch_normalization_2/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/addAdd8batch_normalization_2/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
ќ
Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance
М
Ttraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
к
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_2/moments/variance
Ц
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_2Shape/batch_normalization_2/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_3Const*%
valueB"             *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
”
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
о
Mtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
С
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

DstT0
и
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*/
_output_shapes
:€€€€€€€€€@@ 
ж
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeShapeconv2d_2/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1Const*%
valueB"             *B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
dtype0*
_output_shapes
: 
А
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*/
_output_shapes
:€€€€€€€€€@@ *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
д
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/subSubconv2d_2/Relu*batch_normalization_2/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*/
_output_shapes
:€€€€€€€€€@@ *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
Д
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Ь
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
Ъ
Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
†
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
Ч
Ytraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*&
_output_shapes
: 
™
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*&
_output_shapes
: 
ћ
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeShapeconv2d_2/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/addAdd4batch_normalization_2/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
њ
Ftraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_2/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *5
_class+
)'loc:@batch_normalization_2/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ќ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_2Shapeconv2d_2/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*%
valueB"             *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean
Џ
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean
ё
Itraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

DstT0
Ў
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*/
_output_shapes
:€€€€€€€€€@@ 
а
"training/RMSprop/gradients/AddN_21AddN[training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/truediv*
N*/
_output_shapes
:€€€€€€€€€@@ *
T0* 
_class
loc:@conv2d_2/Relu
—
6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_21conv2d_2/Relu*
T0* 
_class
loc:@conv2d_2/Relu*/
_output_shapes
:€€€€€€€€€@@ 
д
<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
к
;training/RMSprop/gradients/conv2d_2/convolution_grad/ShapeNShapeN batch_normalization_1/cond/Mergeconv2d_2/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_2/convolution_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             *'
_class
loc:@conv2d_2/convolution
љ
Htraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
њ
Itraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter batch_normalization_1/cond/Merge:training/RMSprop/gradients/conv2d_2/convolution_grad/Const6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ƒ
Jtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitchHtraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput"batch_normalization_1/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0*'
_class
loc:@conv2d_2/convolution
щ
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_1/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
И
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
М
$training/RMSprop/gradients/Switch_16Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Ѕ
#training/RMSprop/gradients/Shape_17Shape$training/RMSprop/gradients/Switch_16*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
®
)training/RMSprop/gradients/zeros_16/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
dtype0*
_output_shapes
: 
Г
#training/RMSprop/gradients/zeros_16Fill#training/RMSprop/gradients/Shape_17)training/RMSprop/gradients/zeros_16/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
 
Mtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradMerge#training/RMSprop/gradients/zeros_16Ltraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
А
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_1/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
И
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ў
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
М
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
ч
Ttraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
Л
Ltraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeShape%batch_normalization_1/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
ъ
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:
№
$training/RMSprop/gradients/Switch_17Switchconv2d_1/Relu"batch_normalization_1/cond/pred_id*
T0* 
_class
loc:@conv2d_1/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Ђ
#training/RMSprop/gradients/Shape_18Shape&training/RMSprop/gradients/Switch_17:1*
T0*
out_type0* 
_class
loc:@conv2d_1/Relu*
_output_shapes
:
Р
)training/RMSprop/gradients/zeros_17/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2d_1/Relu
л
#training/RMSprop/gradients/zeros_17Fill#training/RMSprop/gradients/Shape_18)training/RMSprop/gradients/zeros_17/Const*
T0*

index_type0* 
_class
loc:@conv2d_1/Relu*1
_output_shapes
:€€€€€€€€€АА
∆
[training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape#training/RMSprop/gradients/zeros_17*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0* 
_class
loc:@conv2d_1/Relu
Ќ
$training/RMSprop/gradients/Switch_18Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
Є
#training/RMSprop/gradients/Shape_19Shape&training/RMSprop/gradients/Switch_18:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_18/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
б
#training/RMSprop/gradients/zeros_18Fill#training/RMSprop/gradients/Shape_19)training/RMSprop/gradients/zeros_18/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
Љ
Ytraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_18*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*
_output_shapes

:: 
±
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes
:
Љ
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg1batch_normalization_1/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes
:
“
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapeconv2d_1/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
—
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1Const*
valueB:*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
Ї
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape#batch_normalization_1/batchnorm/mul*1
_output_shapes
:€€€€€€€€€АА*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
т
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
¶
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Mulconv2d_1/ReluMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ш
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
г
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
ь
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/sub*
_output_shapes
:
њ
"training/RMSprop/gradients/AddN_22AddNTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
N*
_output_shapes
:
К
Ltraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_22/batch_normalization_1/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes
:
З
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_22*batch_normalization_1/cond/batchnorm/Rsqrt*
_output_shapes
:*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul
≥
"training/RMSprop/gradients/AddN_23AddNYtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_1/beta
Э
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/Neg#batch_normalization_1/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes
:
°
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/Neg%batch_normalization_1/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes
:
ѕ
$training/RMSprop/gradients/Switch_19Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
є
#training/RMSprop/gradients/Shape_20Shape&training/RMSprop/gradients/Switch_19:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
Ю
)training/RMSprop/gradients/zeros_19/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
в
#training/RMSprop/gradients/zeros_19Fill#training/RMSprop/gradients/Shape_20)training/RMSprop/gradients/zeros_19/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
Ј
Ytraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_19*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*
_output_shapes

:: 
ё
Ktraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ShapeConst*%
valueB"            *8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*&
_output_shapes
:
∞
"training/RMSprop/gradients/AddN_24AddNOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
N*
_output_shapes
:
с
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_24 batch_normalization_1/gamma/read*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul
ш
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_24%batch_normalization_1/batchnorm/Rsqrt*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul
Ђ
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_1/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/Rsqrt*
_output_shapes
:
Ѓ
"training/RMSprop/gradients/AddN_25AddNYtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*
_output_shapes
:
Ћ
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
∆
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgs*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
’
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape*
_output_shapes
:*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
ц
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
:
„
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ShapeConst*%
valueB"            *:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*&
_output_shapes
:
ц
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeShape/batch_normalization_1/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
Є
Jtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/addAdd8batch_normalization_1/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
ѕ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *9
_class/
-+loc:@batch_normalization_1/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:*

Tidx0
ќ
Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
и
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
м
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*%
valueB"            *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
”
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
о
Mtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*9
_class/
-+loc:@batch_normalization_1/moments/variance
к
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*1
_output_shapes
:€€€€€€€€€АА
ж
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeShapeconv2d_1/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
ф
Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"            *B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
ѓ
etraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
dtype0*
_output_shapes
: 
В
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
ж
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubconv2d_1/Relu*batch_normalization_1/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*1
_output_shapes
:€€€€€€€€€АА*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
Ж
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ь
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
Ь
Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
†
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*&
_output_shapes
:
™
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*&
_output_shapes
:
ћ
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeShapeconv2d_1/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean
ј
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean
®
Ftraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/addAdd4batch_normalization_1/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *5
_class+
)'loc:@batch_normalization_1/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_1/moments/mean
∆
Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean
ф
Ptraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
…
Ktraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
ќ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_2Shapeconv2d_1/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_3Const*%
valueB"            *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
Е
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

DstT0
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*1
_output_shapes
:€€€€€€€€€АА
в
"training/RMSprop/gradients/AddN_26AddN[training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_1/Relu*
N*1
_output_shapes
:€€€€€€€€€АА
”
6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_26conv2d_1/Relu*
T0* 
_class
loc:@conv2d_1/Relu*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_1/BiasAdd
Ў
;training/RMSprop/gradients/conv2d_1/convolution_grad/ShapeNShapeNconv2d_1_inputconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_1/convolution_grad/ConstConst*%
valueB"            *'
_class
loc:@conv2d_1/convolution*
dtype0*
_output_shapes
:
љ
Htraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
≠
Itraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/RMSprop/gradients/conv2d_1/convolution_grad/Const6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
{
training/RMSprop/ConstConst*%
valueB*    *
dtype0*&
_output_shapes
:
Э
training/RMSprop/Variable
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
е
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(*&
_output_shapes
:*
use_locking(
§
training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable*
T0*,
_class"
 loc:@training/RMSprop/Variable*&
_output_shapes
:
e
training/RMSprop/Const_1Const*
valueB*    *
dtype0*
_output_shapes
:
З
training/RMSprop/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
б
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes
:
Ю
 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
_output_shapes
:*
T0*.
_class$
" loc:@training/RMSprop/Variable_1
e
training/RMSprop/Const_2Const*
valueB*    *
dtype0*
_output_shapes
:
З
training/RMSprop/Variable_2
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
б
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
:
e
training/RMSprop/Const_3Const*
valueB*    *
dtype0*
_output_shapes
:
З
training/RMSprop/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
б
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:
Ю
 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
_output_shapes
:*
T0*.
_class$
" loc:@training/RMSprop/Variable_3
}
training/RMSprop/Const_4Const*
dtype0*&
_output_shapes
: *%
valueB *    
Я
training/RMSprop/Variable_4
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
н
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
validate_shape(*&
_output_shapes
: *
use_locking(
™
 training/RMSprop/Variable_4/readIdentitytraining/RMSprop/Variable_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*&
_output_shapes
: 
e
training/RMSprop/Const_5Const*
valueB *    *
dtype0*
_output_shapes
: 
З
training/RMSprop/Variable_5
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
б
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
validate_shape(*
_output_shapes
: *
use_locking(
Ю
 training/RMSprop/Variable_5/readIdentitytraining/RMSprop/Variable_5*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
e
training/RMSprop/Const_6Const*
valueB *    *
dtype0*
_output_shapes
: 
З
training/RMSprop/Variable_6
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
б
"training/RMSprop/Variable_6/AssignAssigntraining/RMSprop/Variable_6training/RMSprop/Const_6*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_6*
validate_shape(*
_output_shapes
: 
Ю
 training/RMSprop/Variable_6/readIdentitytraining/RMSprop/Variable_6*
T0*.
_class$
" loc:@training/RMSprop/Variable_6*
_output_shapes
: 
e
training/RMSprop/Const_7Const*
dtype0*
_output_shapes
: *
valueB *    
З
training/RMSprop/Variable_7
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
б
"training/RMSprop/Variable_7/AssignAssigntraining/RMSprop/Variable_7training/RMSprop/Const_7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_7
Ю
 training/RMSprop/Variable_7/readIdentitytraining/RMSprop/Variable_7*
_output_shapes
: *
T0*.
_class$
" loc:@training/RMSprop/Variable_7
}
training/RMSprop/Const_8Const*
dtype0*&
_output_shapes
: @*%
valueB @*    
Я
training/RMSprop/Variable_8
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
н
"training/RMSprop/Variable_8/AssignAssigntraining/RMSprop/Variable_8training/RMSprop/Const_8*
T0*.
_class$
" loc:@training/RMSprop/Variable_8*
validate_shape(*&
_output_shapes
: @*
use_locking(
™
 training/RMSprop/Variable_8/readIdentitytraining/RMSprop/Variable_8*&
_output_shapes
: @*
T0*.
_class$
" loc:@training/RMSprop/Variable_8
e
training/RMSprop/Const_9Const*
valueB@*    *
dtype0*
_output_shapes
:@
З
training/RMSprop/Variable_9
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
б
"training/RMSprop/Variable_9/AssignAssigntraining/RMSprop/Variable_9training/RMSprop/Const_9*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_9
Ю
 training/RMSprop/Variable_9/readIdentitytraining/RMSprop/Variable_9*
T0*.
_class$
" loc:@training/RMSprop/Variable_9*
_output_shapes
:@
f
training/RMSprop/Const_10Const*
valueB@*    *
dtype0*
_output_shapes
:@
И
training/RMSprop/Variable_10
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
е
#training/RMSprop/Variable_10/AssignAssigntraining/RMSprop/Variable_10training/RMSprop/Const_10*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10*
validate_shape(*
_output_shapes
:@*
use_locking(
°
!training/RMSprop/Variable_10/readIdentitytraining/RMSprop/Variable_10*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10*
_output_shapes
:@
f
training/RMSprop/Const_11Const*
valueB@*    *
dtype0*
_output_shapes
:@
И
training/RMSprop/Variable_11
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
е
#training/RMSprop/Variable_11/AssignAssigntraining/RMSprop/Variable_11training/RMSprop/Const_11*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11*
validate_shape(*
_output_shapes
:@
°
!training/RMSprop/Variable_11/readIdentitytraining/RMSprop/Variable_11*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11*
_output_shapes
:@
~
training/RMSprop/Const_12Const*
dtype0*&
_output_shapes
:@ *%
valueB@ *    
†
training/RMSprop/Variable_12
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
с
#training/RMSprop/Variable_12/AssignAssigntraining/RMSprop/Variable_12training/RMSprop/Const_12*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12*
validate_shape(*&
_output_shapes
:@ 
≠
!training/RMSprop/Variable_12/readIdentitytraining/RMSprop/Variable_12*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12*&
_output_shapes
:@ 
f
training/RMSprop/Const_13Const*
valueB *    *
dtype0*
_output_shapes
: 
И
training/RMSprop/Variable_13
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
е
#training/RMSprop/Variable_13/AssignAssigntraining/RMSprop/Variable_13training/RMSprop/Const_13*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_13*
validate_shape(*
_output_shapes
: 
°
!training/RMSprop/Variable_13/readIdentitytraining/RMSprop/Variable_13*
_output_shapes
: *
T0*/
_class%
#!loc:@training/RMSprop/Variable_13
f
training/RMSprop/Const_14Const*
valueB *    *
dtype0*
_output_shapes
: 
И
training/RMSprop/Variable_14
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
е
#training/RMSprop/Variable_14/AssignAssigntraining/RMSprop/Variable_14training/RMSprop/Const_14*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
validate_shape(*
_output_shapes
: *
use_locking(
°
!training/RMSprop/Variable_14/readIdentitytraining/RMSprop/Variable_14*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
_output_shapes
: 
f
training/RMSprop/Const_15Const*
dtype0*
_output_shapes
: *
valueB *    
И
training/RMSprop/Variable_15
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
е
#training/RMSprop/Variable_15/AssignAssigntraining/RMSprop/Variable_15training/RMSprop/Const_15*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_15*
validate_shape(*
_output_shapes
: 
°
!training/RMSprop/Variable_15/readIdentitytraining/RMSprop/Variable_15*
_output_shapes
: *
T0*/
_class%
#!loc:@training/RMSprop/Variable_15
~
training/RMSprop/Const_16Const*
dtype0*&
_output_shapes
: *%
valueB *    
†
training/RMSprop/Variable_16
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
с
#training/RMSprop/Variable_16/AssignAssigntraining/RMSprop/Variable_16training/RMSprop/Const_16*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_16*
validate_shape(*&
_output_shapes
: 
≠
!training/RMSprop/Variable_16/readIdentitytraining/RMSprop/Variable_16*&
_output_shapes
: *
T0*/
_class%
#!loc:@training/RMSprop/Variable_16
f
training/RMSprop/Const_17Const*
dtype0*
_output_shapes
:*
valueB*    
И
training/RMSprop/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
е
#training/RMSprop/Variable_17/AssignAssigntraining/RMSprop/Variable_17training/RMSprop/Const_17*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17*
validate_shape(*
_output_shapes
:
°
!training/RMSprop/Variable_17/readIdentitytraining/RMSprop/Variable_17*
_output_shapes
:*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17
f
training/RMSprop/Const_18Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_18
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
#training/RMSprop/Variable_18/AssignAssigntraining/RMSprop/Variable_18training/RMSprop/Const_18*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18*
validate_shape(*
_output_shapes
:*
use_locking(
°
!training/RMSprop/Variable_18/readIdentitytraining/RMSprop/Variable_18*
_output_shapes
:*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18
f
training/RMSprop/Const_19Const*
dtype0*
_output_shapes
:*
valueB*    
И
training/RMSprop/Variable_19
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
#training/RMSprop/Variable_19/AssignAssigntraining/RMSprop/Variable_19training/RMSprop/Const_19*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19*
validate_shape(*
_output_shapes
:
°
!training/RMSprop/Variable_19/readIdentitytraining/RMSprop/Variable_19*
_output_shapes
:*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19
~
training/RMSprop/Const_20Const*%
valueB*    *
dtype0*&
_output_shapes
:
†
training/RMSprop/Variable_20
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
с
#training/RMSprop/Variable_20/AssignAssigntraining/RMSprop/Variable_20training/RMSprop/Const_20*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*
validate_shape(*&
_output_shapes
:
≠
!training/RMSprop/Variable_20/readIdentitytraining/RMSprop/Variable_20*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*&
_output_shapes
:
f
training/RMSprop/Const_21Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_21
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
е
#training/RMSprop/Variable_21/AssignAssigntraining/RMSprop/Variable_21training/RMSprop/Const_21*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21*
validate_shape(*
_output_shapes
:
°
!training/RMSprop/Variable_21/readIdentitytraining/RMSprop/Variable_21*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21*
_output_shapes
:
b
 training/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Є
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: *
use_locking( 
~
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read*&
_output_shapes
:*
T0
[
training/RMSprop/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
_output_shapes
: *
T0
Э
training/RMSprop/SquareSquareItraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
}
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0*&
_output_shapes
:
z
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1*&
_output_shapes
:*
T0
Џ
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(*&
_output_shapes
:
™
training/RMSprop/mul_2MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
^
training/RMSprop/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_23Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
У
&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_23*&
_output_shapes
:*
T0
Э
training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const_22*&
_output_shapes
:*
T0
n
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0*&
_output_shapes
:
]
training/RMSprop/add_1/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 

training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y*
T0*&
_output_shapes
:
Д
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1*
T0*&
_output_shapes
:
~
training/RMSprop/sub_1Subconv2d_1/kernel/readtraining/RMSprop/truediv*
T0*&
_output_shapes
:
 
training/RMSprop/Assign_1Assignconv2d_1/kerneltraining/RMSprop/sub_1*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
v
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes
:
]
training/RMSprop/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 
Ж
training/RMSprop/Square_1Square<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
u
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes
:
r
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
_output_shapes
:*
T0
÷
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
С
training/RMSprop/mul_5MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_25Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_25*
T0*
_output_shapes
:
Х
 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_24*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
T0*
_output_shapes
:
]
training/RMSprop/add_3/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
T0*
_output_shapes
:
z
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes
:
r
training/RMSprop/sub_3Subconv2d_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes
:
Ї
training/RMSprop/Assign_3Assignconv2d_1/biastraining/RMSprop/sub_3*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:
v
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes
:
]
training/RMSprop/sub_4/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_2Square"training/RMSprop/gradients/AddN_25*
T0*
_output_shapes
:
u
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes
:
r
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
validate_shape(*
_output_shapes
:*
use_locking(
w
training/RMSprop/mul_8MulRMSprop/lr/read"training/RMSprop/gradients/AddN_25*
T0*
_output_shapes
:
^
training/RMSprop/Const_26Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_27Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_27*
_output_shapes
:*
T0
Х
 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_26*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes
:
]
training/RMSprop/add_5/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
T0*
_output_shapes
:
z
training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
_output_shapes
:*
T0
А
training/RMSprop/sub_5Sub batch_normalization_1/gamma/readtraining/RMSprop/truediv_2*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_5Assignbatch_normalization_1/gammatraining/RMSprop/sub_5*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
_output_shapes
:*
T0
]
training/RMSprop/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_3Square"training/RMSprop/gradients/AddN_23*
T0*
_output_shapes
:
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
x
training/RMSprop/mul_11MulRMSprop/lr/read"training/RMSprop/gradients/AddN_23*
T0*
_output_shapes
:
^
training/RMSprop/Const_28Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_29Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Л
(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_29*
T0*
_output_shapes
:
Х
 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_28*
_output_shapes
:*
T0
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:
]
training/RMSprop/add_7/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
_output_shapes
:*
T0
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
T0*
_output_shapes
:

training/RMSprop/sub_7Subbatch_normalization_1/beta/readtraining/RMSprop/truediv_3*
_output_shapes
:*
T0
‘
training/RMSprop/Assign_7Assignbatch_normalization_1/betatraining/RMSprop/sub_7*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:
Г
training/RMSprop/mul_12MulRMSprop/rho/read training/RMSprop/Variable_4/read*&
_output_shapes
: *
T0
]
training/RMSprop/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
T0*
_output_shapes
: 
Я
training/RMSprop/Square_4SquareItraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
В
training/RMSprop/mul_13Multraining/RMSprop/sub_8training/RMSprop/Square_4*
T0*&
_output_shapes
: 
А
training/RMSprop/add_8Addtraining/RMSprop/mul_12training/RMSprop/mul_13*
T0*&
_output_shapes
: 
в
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
validate_shape(*&
_output_shapes
: *
use_locking(
Ђ
training/RMSprop/mul_14MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
^
training/RMSprop/Const_30Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_31Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Ч
(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_31*
T0*&
_output_shapes
: 
°
 training/RMSprop/clip_by_value_4Maximum(training/RMSprop/clip_by_value_4/Minimumtraining/RMSprop/Const_30*
T0*&
_output_shapes
: 
r
training/RMSprop/Sqrt_4Sqrt training/RMSprop/clip_by_value_4*&
_output_shapes
: *
T0
]
training/RMSprop/add_9/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Б
training/RMSprop/add_9Addtraining/RMSprop/Sqrt_4training/RMSprop/add_9/y*&
_output_shapes
: *
T0
З
training/RMSprop/truediv_4RealDivtraining/RMSprop/mul_14training/RMSprop/add_9*&
_output_shapes
: *
T0
А
training/RMSprop/sub_9Subconv2d_2/kernel/readtraining/RMSprop/truediv_4*&
_output_shapes
: *
T0
 
training/RMSprop/Assign_9Assignconv2d_2/kerneltraining/RMSprop/sub_9*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel
w
training/RMSprop/mul_15MulRMSprop/rho/read training/RMSprop/Variable_5/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 
Ж
training/RMSprop/Square_5Square<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
w
training/RMSprop/mul_16Multraining/RMSprop/sub_10training/RMSprop/Square_5*
T0*
_output_shapes
: 
u
training/RMSprop/add_10Addtraining/RMSprop/mul_15training/RMSprop/mul_16*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
validate_shape(*
_output_shapes
: 
Т
training/RMSprop/mul_17MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
^
training/RMSprop/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_33Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_33*
T0*
_output_shapes
: 
Х
 training/RMSprop/clip_by_value_5Maximum(training/RMSprop/clip_by_value_5/Minimumtraining/RMSprop/Const_32*
T0*
_output_shapes
: 
f
training/RMSprop/Sqrt_5Sqrt training/RMSprop/clip_by_value_5*
T0*
_output_shapes
: 
^
training/RMSprop/add_11/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_11Addtraining/RMSprop/Sqrt_5training/RMSprop/add_11/y*
_output_shapes
: *
T0
|
training/RMSprop/truediv_5RealDivtraining/RMSprop/mul_17training/RMSprop/add_11*
T0*
_output_shapes
: 
s
training/RMSprop/sub_11Subconv2d_2/bias/readtraining/RMSprop/truediv_5*
T0*
_output_shapes
: 
Љ
training/RMSprop/Assign_11Assignconv2d_2/biastraining/RMSprop/sub_11*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
w
training/RMSprop/mul_18MulRMSprop/rho/read training/RMSprop/Variable_6/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_12/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_12Subtraining/RMSprop/sub_12/xRMSprop/rho/read*
_output_shapes
: *
T0
l
training/RMSprop/Square_6Square"training/RMSprop/gradients/AddN_20*
T0*
_output_shapes
: 
w
training/RMSprop/mul_19Multraining/RMSprop/sub_12training/RMSprop/Square_6*
T0*
_output_shapes
: 
u
training/RMSprop/add_12Addtraining/RMSprop/mul_18training/RMSprop/mul_19*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_12Assigntraining/RMSprop/Variable_6training/RMSprop/add_12*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_6
x
training/RMSprop/mul_20MulRMSprop/lr/read"training/RMSprop/gradients/AddN_20*
T0*
_output_shapes
: 
^
training/RMSprop/Const_34Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_35Const*
dtype0*
_output_shapes
: *
valueB
 *  А
М
(training/RMSprop/clip_by_value_6/MinimumMinimumtraining/RMSprop/add_12training/RMSprop/Const_35*
_output_shapes
: *
T0
Х
 training/RMSprop/clip_by_value_6Maximum(training/RMSprop/clip_by_value_6/Minimumtraining/RMSprop/Const_34*
_output_shapes
: *
T0
f
training/RMSprop/Sqrt_6Sqrt training/RMSprop/clip_by_value_6*
T0*
_output_shapes
: 
^
training/RMSprop/add_13/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_13Addtraining/RMSprop/Sqrt_6training/RMSprop/add_13/y*
_output_shapes
: *
T0
|
training/RMSprop/truediv_6RealDivtraining/RMSprop/mul_20training/RMSprop/add_13*
T0*
_output_shapes
: 
Б
training/RMSprop/sub_13Sub batch_normalization_2/gamma/readtraining/RMSprop/truediv_6*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_13Assignbatch_normalization_2/gammatraining/RMSprop/sub_13*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
w
training/RMSprop/mul_21MulRMSprop/rho/read training/RMSprop/Variable_7/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_14/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_14Subtraining/RMSprop/sub_14/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_7Square"training/RMSprop/gradients/AddN_18*
T0*
_output_shapes
: 
w
training/RMSprop/mul_22Multraining/RMSprop/sub_14training/RMSprop/Square_7*
_output_shapes
: *
T0
u
training/RMSprop/add_14Addtraining/RMSprop/mul_21training/RMSprop/mul_22*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_14Assigntraining/RMSprop/Variable_7training/RMSprop/add_14*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_7*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_23MulRMSprop/lr/read"training/RMSprop/gradients/AddN_18*
T0*
_output_shapes
: 
^
training/RMSprop/Const_36Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_37Const*
dtype0*
_output_shapes
: *
valueB
 *  А
М
(training/RMSprop/clip_by_value_7/MinimumMinimumtraining/RMSprop/add_14training/RMSprop/Const_37*
_output_shapes
: *
T0
Х
 training/RMSprop/clip_by_value_7Maximum(training/RMSprop/clip_by_value_7/Minimumtraining/RMSprop/Const_36*
T0*
_output_shapes
: 
f
training/RMSprop/Sqrt_7Sqrt training/RMSprop/clip_by_value_7*
_output_shapes
: *
T0
^
training/RMSprop/add_15/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_15Addtraining/RMSprop/Sqrt_7training/RMSprop/add_15/y*
T0*
_output_shapes
: 
|
training/RMSprop/truediv_7RealDivtraining/RMSprop/mul_23training/RMSprop/add_15*
_output_shapes
: *
T0
А
training/RMSprop/sub_15Subbatch_normalization_2/beta/readtraining/RMSprop/truediv_7*
T0*
_output_shapes
: 
÷
training/RMSprop/Assign_15Assignbatch_normalization_2/betatraining/RMSprop/sub_15*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
: 
Г
training/RMSprop/mul_24MulRMSprop/rho/read training/RMSprop/Variable_8/read*
T0*&
_output_shapes
: @
^
training/RMSprop/sub_16/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_16Subtraining/RMSprop/sub_16/xRMSprop/rho/read*
T0*
_output_shapes
: 
Я
training/RMSprop/Square_8SquareItraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
Г
training/RMSprop/mul_25Multraining/RMSprop/sub_16training/RMSprop/Square_8*&
_output_shapes
: @*
T0
Б
training/RMSprop/add_16Addtraining/RMSprop/mul_24training/RMSprop/mul_25*
T0*&
_output_shapes
: @
д
training/RMSprop/Assign_16Assigntraining/RMSprop/Variable_8training/RMSprop/add_16*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_8
Ђ
training/RMSprop/mul_26MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
^
training/RMSprop/Const_38Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_39Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Ш
(training/RMSprop/clip_by_value_8/MinimumMinimumtraining/RMSprop/add_16training/RMSprop/Const_39*
T0*&
_output_shapes
: @
°
 training/RMSprop/clip_by_value_8Maximum(training/RMSprop/clip_by_value_8/Minimumtraining/RMSprop/Const_38*
T0*&
_output_shapes
: @
r
training/RMSprop/Sqrt_8Sqrt training/RMSprop/clip_by_value_8*&
_output_shapes
: @*
T0
^
training/RMSprop/add_17/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Г
training/RMSprop/add_17Addtraining/RMSprop/Sqrt_8training/RMSprop/add_17/y*
T0*&
_output_shapes
: @
И
training/RMSprop/truediv_8RealDivtraining/RMSprop/mul_26training/RMSprop/add_17*&
_output_shapes
: @*
T0
Б
training/RMSprop/sub_17Subconv2d_3/kernel/readtraining/RMSprop/truediv_8*&
_output_shapes
: @*
T0
ћ
training/RMSprop/Assign_17Assignconv2d_3/kerneltraining/RMSprop/sub_17*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
: @
w
training/RMSprop/mul_27MulRMSprop/rho/read training/RMSprop/Variable_9/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_18/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_18Subtraining/RMSprop/sub_18/xRMSprop/rho/read*
_output_shapes
: *
T0
Ж
training/RMSprop/Square_9Square<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
w
training/RMSprop/mul_28Multraining/RMSprop/sub_18training/RMSprop/Square_9*
T0*
_output_shapes
:@
u
training/RMSprop/add_18Addtraining/RMSprop/mul_27training/RMSprop/mul_28*
T0*
_output_shapes
:@
Ў
training/RMSprop/Assign_18Assigntraining/RMSprop/Variable_9training/RMSprop/add_18*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_9*
validate_shape(*
_output_shapes
:@
Т
training/RMSprop/mul_29MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
^
training/RMSprop/Const_40Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_41Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_9/MinimumMinimumtraining/RMSprop/add_18training/RMSprop/Const_41*
_output_shapes
:@*
T0
Х
 training/RMSprop/clip_by_value_9Maximum(training/RMSprop/clip_by_value_9/Minimumtraining/RMSprop/Const_40*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_9Sqrt training/RMSprop/clip_by_value_9*
T0*
_output_shapes
:@
^
training/RMSprop/add_19/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_19Addtraining/RMSprop/Sqrt_9training/RMSprop/add_19/y*
T0*
_output_shapes
:@
|
training/RMSprop/truediv_9RealDivtraining/RMSprop/mul_29training/RMSprop/add_19*
_output_shapes
:@*
T0
s
training/RMSprop/sub_19Subconv2d_3/bias/readtraining/RMSprop/truediv_9*
_output_shapes
:@*
T0
Љ
training/RMSprop/Assign_19Assignconv2d_3/biastraining/RMSprop/sub_19*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:@
x
training/RMSprop/mul_30MulRMSprop/rho/read!training/RMSprop/Variable_10/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_20Subtraining/RMSprop/sub_20/xRMSprop/rho/read*
T0*
_output_shapes
: 
m
training/RMSprop/Square_10Square"training/RMSprop/gradients/AddN_15*
T0*
_output_shapes
:@
x
training/RMSprop/mul_31Multraining/RMSprop/sub_20training/RMSprop/Square_10*
T0*
_output_shapes
:@
u
training/RMSprop/add_20Addtraining/RMSprop/mul_30training/RMSprop/mul_31*
_output_shapes
:@*
T0
Џ
training/RMSprop/Assign_20Assigntraining/RMSprop/Variable_10training/RMSprop/add_20*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10
x
training/RMSprop/mul_32MulRMSprop/lr/read"training/RMSprop/gradients/AddN_15*
T0*
_output_shapes
:@
^
training/RMSprop/Const_42Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_43Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_10/MinimumMinimumtraining/RMSprop/add_20training/RMSprop/Const_43*
T0*
_output_shapes
:@
Ч
!training/RMSprop/clip_by_value_10Maximum)training/RMSprop/clip_by_value_10/Minimumtraining/RMSprop/Const_42*
T0*
_output_shapes
:@
h
training/RMSprop/Sqrt_10Sqrt!training/RMSprop/clip_by_value_10*
T0*
_output_shapes
:@
^
training/RMSprop/add_21/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_21Addtraining/RMSprop/Sqrt_10training/RMSprop/add_21/y*
T0*
_output_shapes
:@
}
training/RMSprop/truediv_10RealDivtraining/RMSprop/mul_32training/RMSprop/add_21*
T0*
_output_shapes
:@
В
training/RMSprop/sub_21Sub batch_normalization_3/gamma/readtraining/RMSprop/truediv_10*
_output_shapes
:@*
T0
Ў
training/RMSprop/Assign_21Assignbatch_normalization_3/gammatraining/RMSprop/sub_21*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:@
x
training/RMSprop/mul_33MulRMSprop/rho/read!training/RMSprop/Variable_11/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_22/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_22Subtraining/RMSprop/sub_22/xRMSprop/rho/read*
_output_shapes
: *
T0
m
training/RMSprop/Square_11Square"training/RMSprop/gradients/AddN_13*
T0*
_output_shapes
:@
x
training/RMSprop/mul_34Multraining/RMSprop/sub_22training/RMSprop/Square_11*
T0*
_output_shapes
:@
u
training/RMSprop/add_22Addtraining/RMSprop/mul_33training/RMSprop/mul_34*
_output_shapes
:@*
T0
Џ
training/RMSprop/Assign_22Assigntraining/RMSprop/Variable_11training/RMSprop/add_22*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11*
validate_shape(*
_output_shapes
:@*
use_locking(
x
training/RMSprop/mul_35MulRMSprop/lr/read"training/RMSprop/gradients/AddN_13*
T0*
_output_shapes
:@
^
training/RMSprop/Const_44Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_45Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_11/MinimumMinimumtraining/RMSprop/add_22training/RMSprop/Const_45*
T0*
_output_shapes
:@
Ч
!training/RMSprop/clip_by_value_11Maximum)training/RMSprop/clip_by_value_11/Minimumtraining/RMSprop/Const_44*
T0*
_output_shapes
:@
h
training/RMSprop/Sqrt_11Sqrt!training/RMSprop/clip_by_value_11*
T0*
_output_shapes
:@
^
training/RMSprop/add_23/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_23Addtraining/RMSprop/Sqrt_11training/RMSprop/add_23/y*
T0*
_output_shapes
:@
}
training/RMSprop/truediv_11RealDivtraining/RMSprop/mul_35training/RMSprop/add_23*
T0*
_output_shapes
:@
Б
training/RMSprop/sub_23Subbatch_normalization_3/beta/readtraining/RMSprop/truediv_11*
T0*
_output_shapes
:@
÷
training/RMSprop/Assign_23Assignbatch_normalization_3/betatraining/RMSprop/sub_23*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:@
Д
training/RMSprop/mul_36MulRMSprop/rho/read!training/RMSprop/Variable_12/read*
T0*&
_output_shapes
:@ 
^
training/RMSprop/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_24Subtraining/RMSprop/sub_24/xRMSprop/rho/read*
T0*
_output_shapes
: 
†
training/RMSprop/Square_12SquareItraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
Д
training/RMSprop/mul_37Multraining/RMSprop/sub_24training/RMSprop/Square_12*&
_output_shapes
:@ *
T0
Б
training/RMSprop/add_24Addtraining/RMSprop/mul_36training/RMSprop/mul_37*&
_output_shapes
:@ *
T0
ж
training/RMSprop/Assign_24Assigntraining/RMSprop/Variable_12training/RMSprop/add_24*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12
Ђ
training/RMSprop/mul_38MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
^
training/RMSprop/Const_46Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_47Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_12/MinimumMinimumtraining/RMSprop/add_24training/RMSprop/Const_47*
T0*&
_output_shapes
:@ 
£
!training/RMSprop/clip_by_value_12Maximum)training/RMSprop/clip_by_value_12/Minimumtraining/RMSprop/Const_46*&
_output_shapes
:@ *
T0
t
training/RMSprop/Sqrt_12Sqrt!training/RMSprop/clip_by_value_12*
T0*&
_output_shapes
:@ 
^
training/RMSprop/add_25/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_25Addtraining/RMSprop/Sqrt_12training/RMSprop/add_25/y*
T0*&
_output_shapes
:@ 
Й
training/RMSprop/truediv_12RealDivtraining/RMSprop/mul_38training/RMSprop/add_25*&
_output_shapes
:@ *
T0
В
training/RMSprop/sub_25Subconv2d_4/kernel/readtraining/RMSprop/truediv_12*
T0*&
_output_shapes
:@ 
ћ
training/RMSprop/Assign_25Assignconv2d_4/kerneltraining/RMSprop/sub_25*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:@ 
x
training/RMSprop/mul_39MulRMSprop/rho/read!training/RMSprop/Variable_13/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_26/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_26Subtraining/RMSprop/sub_26/xRMSprop/rho/read*
T0*
_output_shapes
: 
З
training/RMSprop/Square_13Square<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
x
training/RMSprop/mul_40Multraining/RMSprop/sub_26training/RMSprop/Square_13*
T0*
_output_shapes
: 
u
training/RMSprop/add_26Addtraining/RMSprop/mul_39training/RMSprop/mul_40*
_output_shapes
: *
T0
Џ
training/RMSprop/Assign_26Assigntraining/RMSprop/Variable_13training/RMSprop/add_26*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_13*
validate_shape(*
_output_shapes
: 
Т
training/RMSprop/mul_41MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
^
training/RMSprop/Const_48Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_49Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_13/MinimumMinimumtraining/RMSprop/add_26training/RMSprop/Const_49*
_output_shapes
: *
T0
Ч
!training/RMSprop/clip_by_value_13Maximum)training/RMSprop/clip_by_value_13/Minimumtraining/RMSprop/Const_48*
T0*
_output_shapes
: 
h
training/RMSprop/Sqrt_13Sqrt!training/RMSprop/clip_by_value_13*
T0*
_output_shapes
: 
^
training/RMSprop/add_27/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_27Addtraining/RMSprop/Sqrt_13training/RMSprop/add_27/y*
T0*
_output_shapes
: 
}
training/RMSprop/truediv_13RealDivtraining/RMSprop/mul_41training/RMSprop/add_27*
T0*
_output_shapes
: 
t
training/RMSprop/sub_27Subconv2d_4/bias/readtraining/RMSprop/truediv_13*
_output_shapes
: *
T0
Љ
training/RMSprop/Assign_27Assignconv2d_4/biastraining/RMSprop/sub_27*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_42MulRMSprop/rho/read!training/RMSprop/Variable_14/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_28/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_28Subtraining/RMSprop/sub_28/xRMSprop/rho/read*
T0*
_output_shapes
: 
m
training/RMSprop/Square_14Square"training/RMSprop/gradients/AddN_10*
T0*
_output_shapes
: 
x
training/RMSprop/mul_43Multraining/RMSprop/sub_28training/RMSprop/Square_14*
T0*
_output_shapes
: 
u
training/RMSprop/add_28Addtraining/RMSprop/mul_42training/RMSprop/mul_43*
T0*
_output_shapes
: 
Џ
training/RMSprop/Assign_28Assigntraining/RMSprop/Variable_14training/RMSprop/add_28*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
validate_shape(*
_output_shapes
: *
use_locking(
x
training/RMSprop/mul_44MulRMSprop/lr/read"training/RMSprop/gradients/AddN_10*
_output_shapes
: *
T0
^
training/RMSprop/Const_50Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_51Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_14/MinimumMinimumtraining/RMSprop/add_28training/RMSprop/Const_51*
_output_shapes
: *
T0
Ч
!training/RMSprop/clip_by_value_14Maximum)training/RMSprop/clip_by_value_14/Minimumtraining/RMSprop/Const_50*
T0*
_output_shapes
: 
h
training/RMSprop/Sqrt_14Sqrt!training/RMSprop/clip_by_value_14*
_output_shapes
: *
T0
^
training/RMSprop/add_29/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_29Addtraining/RMSprop/Sqrt_14training/RMSprop/add_29/y*
T0*
_output_shapes
: 
}
training/RMSprop/truediv_14RealDivtraining/RMSprop/mul_44training/RMSprop/add_29*
T0*
_output_shapes
: 
В
training/RMSprop/sub_29Sub batch_normalization_4/gamma/readtraining/RMSprop/truediv_14*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_29Assignbatch_normalization_4/gammatraining/RMSprop/sub_29*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_45MulRMSprop/rho/read!training/RMSprop/Variable_15/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_30/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_30Subtraining/RMSprop/sub_30/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_15Square!training/RMSprop/gradients/AddN_8*
T0*
_output_shapes
: 
x
training/RMSprop/mul_46Multraining/RMSprop/sub_30training/RMSprop/Square_15*
T0*
_output_shapes
: 
u
training/RMSprop/add_30Addtraining/RMSprop/mul_45training/RMSprop/mul_46*
T0*
_output_shapes
: 
Џ
training/RMSprop/Assign_30Assigntraining/RMSprop/Variable_15training/RMSprop/add_30*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_15*
validate_shape(*
_output_shapes
: 
w
training/RMSprop/mul_47MulRMSprop/lr/read!training/RMSprop/gradients/AddN_8*
T0*
_output_shapes
: 
^
training/RMSprop/Const_52Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_53Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_15/MinimumMinimumtraining/RMSprop/add_30training/RMSprop/Const_53*
_output_shapes
: *
T0
Ч
!training/RMSprop/clip_by_value_15Maximum)training/RMSprop/clip_by_value_15/Minimumtraining/RMSprop/Const_52*
T0*
_output_shapes
: 
h
training/RMSprop/Sqrt_15Sqrt!training/RMSprop/clip_by_value_15*
_output_shapes
: *
T0
^
training/RMSprop/add_31/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_31Addtraining/RMSprop/Sqrt_15training/RMSprop/add_31/y*
_output_shapes
: *
T0
}
training/RMSprop/truediv_15RealDivtraining/RMSprop/mul_47training/RMSprop/add_31*
T0*
_output_shapes
: 
Б
training/RMSprop/sub_31Subbatch_normalization_4/beta/readtraining/RMSprop/truediv_15*
T0*
_output_shapes
: 
÷
training/RMSprop/Assign_31Assignbatch_normalization_4/betatraining/RMSprop/sub_31*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
: 
Д
training/RMSprop/mul_48MulRMSprop/rho/read!training/RMSprop/Variable_16/read*
T0*&
_output_shapes
: 
^
training/RMSprop/sub_32/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_32Subtraining/RMSprop/sub_32/xRMSprop/rho/read*
_output_shapes
: *
T0
†
training/RMSprop/Square_16SquareItraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
Д
training/RMSprop/mul_49Multraining/RMSprop/sub_32training/RMSprop/Square_16*&
_output_shapes
: *
T0
Б
training/RMSprop/add_32Addtraining/RMSprop/mul_48training/RMSprop/mul_49*&
_output_shapes
: *
T0
ж
training/RMSprop/Assign_32Assigntraining/RMSprop/Variable_16training/RMSprop/add_32*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_16*
validate_shape(*&
_output_shapes
: 
Ђ
training/RMSprop/mul_50MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
^
training/RMSprop/Const_54Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_55Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_16/MinimumMinimumtraining/RMSprop/add_32training/RMSprop/Const_55*
T0*&
_output_shapes
: 
£
!training/RMSprop/clip_by_value_16Maximum)training/RMSprop/clip_by_value_16/Minimumtraining/RMSprop/Const_54*
T0*&
_output_shapes
: 
t
training/RMSprop/Sqrt_16Sqrt!training/RMSprop/clip_by_value_16*
T0*&
_output_shapes
: 
^
training/RMSprop/add_33/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_33Addtraining/RMSprop/Sqrt_16training/RMSprop/add_33/y*
T0*&
_output_shapes
: 
Й
training/RMSprop/truediv_16RealDivtraining/RMSprop/mul_50training/RMSprop/add_33*
T0*&
_output_shapes
: 
В
training/RMSprop/sub_33Subconv2d_5/kernel/readtraining/RMSprop/truediv_16*
T0*&
_output_shapes
: 
ћ
training/RMSprop/Assign_33Assignconv2d_5/kerneltraining/RMSprop/sub_33*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel
x
training/RMSprop/mul_51MulRMSprop/rho/read!training/RMSprop/Variable_17/read*
_output_shapes
:*
T0
^
training/RMSprop/sub_34/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_34Subtraining/RMSprop/sub_34/xRMSprop/rho/read*
_output_shapes
: *
T0
З
training/RMSprop/Square_17Square<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
x
training/RMSprop/mul_52Multraining/RMSprop/sub_34training/RMSprop/Square_17*
T0*
_output_shapes
:
u
training/RMSprop/add_34Addtraining/RMSprop/mul_51training/RMSprop/mul_52*
T0*
_output_shapes
:
Џ
training/RMSprop/Assign_34Assigntraining/RMSprop/Variable_17training/RMSprop/add_34*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(
Т
training/RMSprop/mul_53MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_56Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_57Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_17/MinimumMinimumtraining/RMSprop/add_34training/RMSprop/Const_57*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_17Maximum)training/RMSprop/clip_by_value_17/Minimumtraining/RMSprop/Const_56*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_17Sqrt!training/RMSprop/clip_by_value_17*
T0*
_output_shapes
:
^
training/RMSprop/add_35/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_35Addtraining/RMSprop/Sqrt_17training/RMSprop/add_35/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_17RealDivtraining/RMSprop/mul_53training/RMSprop/add_35*
_output_shapes
:*
T0
t
training/RMSprop/sub_35Subconv2d_5/bias/readtraining/RMSprop/truediv_17*
T0*
_output_shapes
:
Љ
training/RMSprop/Assign_35Assignconv2d_5/biastraining/RMSprop/sub_35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias
x
training/RMSprop/mul_54MulRMSprop/rho/read!training/RMSprop/Variable_18/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_36/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_36Subtraining/RMSprop/sub_36/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_18Square!training/RMSprop/gradients/AddN_5*
T0*
_output_shapes
:
x
training/RMSprop/mul_55Multraining/RMSprop/sub_36training/RMSprop/Square_18*
T0*
_output_shapes
:
u
training/RMSprop/add_36Addtraining/RMSprop/mul_54training/RMSprop/mul_55*
_output_shapes
:*
T0
Џ
training/RMSprop/Assign_36Assigntraining/RMSprop/Variable_18training/RMSprop/add_36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18
w
training/RMSprop/mul_56MulRMSprop/lr/read!training/RMSprop/gradients/AddN_5*
T0*
_output_shapes
:
^
training/RMSprop/Const_58Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_59Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_18/MinimumMinimumtraining/RMSprop/add_36training/RMSprop/Const_59*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_18Maximum)training/RMSprop/clip_by_value_18/Minimumtraining/RMSprop/Const_58*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_18Sqrt!training/RMSprop/clip_by_value_18*
T0*
_output_shapes
:
^
training/RMSprop/add_37/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_37Addtraining/RMSprop/Sqrt_18training/RMSprop/add_37/y*
_output_shapes
:*
T0
}
training/RMSprop/truediv_18RealDivtraining/RMSprop/mul_56training/RMSprop/add_37*
_output_shapes
:*
T0
В
training/RMSprop/sub_37Sub batch_normalization_5/gamma/readtraining/RMSprop/truediv_18*
T0*
_output_shapes
:
Ў
training/RMSprop/Assign_37Assignbatch_normalization_5/gammatraining/RMSprop/sub_37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma
x
training/RMSprop/mul_57MulRMSprop/rho/read!training/RMSprop/Variable_19/read*
_output_shapes
:*
T0
^
training/RMSprop/sub_38/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_38Subtraining/RMSprop/sub_38/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_19Square!training/RMSprop/gradients/AddN_3*
T0*
_output_shapes
:
x
training/RMSprop/mul_58Multraining/RMSprop/sub_38training/RMSprop/Square_19*
T0*
_output_shapes
:
u
training/RMSprop/add_38Addtraining/RMSprop/mul_57training/RMSprop/mul_58*
_output_shapes
:*
T0
Џ
training/RMSprop/Assign_38Assigntraining/RMSprop/Variable_19training/RMSprop/add_38*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(
w
training/RMSprop/mul_59MulRMSprop/lr/read!training/RMSprop/gradients/AddN_3*
_output_shapes
:*
T0
^
training/RMSprop/Const_60Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_61Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_19/MinimumMinimumtraining/RMSprop/add_38training/RMSprop/Const_61*
_output_shapes
:*
T0
Ч
!training/RMSprop/clip_by_value_19Maximum)training/RMSprop/clip_by_value_19/Minimumtraining/RMSprop/Const_60*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_19Sqrt!training/RMSprop/clip_by_value_19*
T0*
_output_shapes
:
^
training/RMSprop/add_39/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_39Addtraining/RMSprop/Sqrt_19training/RMSprop/add_39/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_19RealDivtraining/RMSprop/mul_59training/RMSprop/add_39*
T0*
_output_shapes
:
Б
training/RMSprop/sub_39Subbatch_normalization_5/beta/readtraining/RMSprop/truediv_19*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_39Assignbatch_normalization_5/betatraining/RMSprop/sub_39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta
Д
training/RMSprop/mul_60MulRMSprop/rho/read!training/RMSprop/Variable_20/read*&
_output_shapes
:*
T0
^
training/RMSprop/sub_40/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_40Subtraining/RMSprop/sub_40/xRMSprop/rho/read*
T0*
_output_shapes
: 
†
training/RMSprop/Square_20SquareItraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
Д
training/RMSprop/mul_61Multraining/RMSprop/sub_40training/RMSprop/Square_20*
T0*&
_output_shapes
:
Б
training/RMSprop/add_40Addtraining/RMSprop/mul_60training/RMSprop/mul_61*&
_output_shapes
:*
T0
ж
training/RMSprop/Assign_40Assigntraining/RMSprop/Variable_20training/RMSprop/add_40*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*
validate_shape(*&
_output_shapes
:
Ђ
training/RMSprop/mul_62MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
^
training/RMSprop/Const_62Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_63Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_20/MinimumMinimumtraining/RMSprop/add_40training/RMSprop/Const_63*
T0*&
_output_shapes
:
£
!training/RMSprop/clip_by_value_20Maximum)training/RMSprop/clip_by_value_20/Minimumtraining/RMSprop/Const_62*
T0*&
_output_shapes
:
t
training/RMSprop/Sqrt_20Sqrt!training/RMSprop/clip_by_value_20*
T0*&
_output_shapes
:
^
training/RMSprop/add_41/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_41Addtraining/RMSprop/Sqrt_20training/RMSprop/add_41/y*
T0*&
_output_shapes
:
Й
training/RMSprop/truediv_20RealDivtraining/RMSprop/mul_62training/RMSprop/add_41*
T0*&
_output_shapes
:
В
training/RMSprop/sub_41Subconv2d_6/kernel/readtraining/RMSprop/truediv_20*
T0*&
_output_shapes
:
ћ
training/RMSprop/Assign_41Assignconv2d_6/kerneltraining/RMSprop/sub_41*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
x
training/RMSprop/mul_63MulRMSprop/rho/read!training/RMSprop/Variable_21/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_42/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_42Subtraining/RMSprop/sub_42/xRMSprop/rho/read*
_output_shapes
: *
T0
З
training/RMSprop/Square_21Square<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
x
training/RMSprop/mul_64Multraining/RMSprop/sub_42training/RMSprop/Square_21*
_output_shapes
:*
T0
u
training/RMSprop/add_42Addtraining/RMSprop/mul_63training/RMSprop/mul_64*
_output_shapes
:*
T0
Џ
training/RMSprop/Assign_42Assigntraining/RMSprop/Variable_21training/RMSprop/add_42*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21*
validate_shape(*
_output_shapes
:
Т
training/RMSprop/mul_65MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_64Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_65Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_21/MinimumMinimumtraining/RMSprop/add_42training/RMSprop/Const_65*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_21Maximum)training/RMSprop/clip_by_value_21/Minimumtraining/RMSprop/Const_64*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_21Sqrt!training/RMSprop/clip_by_value_21*
_output_shapes
:*
T0
^
training/RMSprop/add_43/yConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
x
training/RMSprop/add_43Addtraining/RMSprop/Sqrt_21training/RMSprop/add_43/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_21RealDivtraining/RMSprop/mul_65training/RMSprop/add_43*
T0*
_output_shapes
:
t
training/RMSprop/sub_43Subconv2d_6/bias/readtraining/RMSprop/truediv_21*
_output_shapes
:*
T0
Љ
training/RMSprop/Assign_43Assignconv2d_6/biastraining/RMSprop/sub_43*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ќ
training/group_depsNoOp	^loss/mul&^batch_normalization_1/AssignMovingAvg(^batch_normalization_1/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11^training/RMSprop/Assign_12^training/RMSprop/Assign_13^training/RMSprop/Assign_14^training/RMSprop/Assign_15^training/RMSprop/Assign_16^training/RMSprop/Assign_17^training/RMSprop/Assign_18^training/RMSprop/Assign_19^training/RMSprop/Assign_20^training/RMSprop/Assign_21^training/RMSprop/Assign_22^training/RMSprop/Assign_23^training/RMSprop/Assign_24^training/RMSprop/Assign_25^training/RMSprop/Assign_26^training/RMSprop/Assign_27^training/RMSprop/Assign_28^training/RMSprop/Assign_29^training/RMSprop/Assign_30^training/RMSprop/Assign_31^training/RMSprop/Assign_32^training/RMSprop/Assign_33^training/RMSprop/Assign_34^training/RMSprop/Assign_35^training/RMSprop/Assign_36^training/RMSprop/Assign_37^training/RMSprop/Assign_38^training/RMSprop/Assign_39^training/RMSprop/Assign_40^training/RMSprop/Assign_41^training/RMSprop/Assign_42^training/RMSprop/Assign_43


group_depsNoOp	^loss/mul
И
IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ґ
IsVariableInitialized_2IsVariableInitializedbatch_normalization_1/gamma*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma
†
IsVariableInitialized_3IsVariableInitializedbatch_normalization_1/beta*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta
Ѓ
IsVariableInitialized_4IsVariableInitialized!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
ґ
IsVariableInitialized_5IsVariableInitialized%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance
К
IsVariableInitialized_6IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_7IsVariableInitializedconv2d_2/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_2/bias
Ґ
IsVariableInitialized_8IsVariableInitializedbatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
†
IsVariableInitialized_9IsVariableInitializedbatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
ѓ
IsVariableInitialized_10IsVariableInitialized!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_2/moving_mean
Ј
IsVariableInitialized_11IsVariableInitialized%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_2/moving_variance
Л
IsVariableInitialized_12IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_13IsVariableInitializedconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_14IsVariableInitializedbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
°
IsVariableInitialized_15IsVariableInitializedbatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
ѓ
IsVariableInitialized_16IsVariableInitialized!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_17IsVariableInitialized%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_18IsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_19IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_20IsVariableInitializedbatch_normalization_4/gamma*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_4/gamma
°
IsVariableInitialized_21IsVariableInitializedbatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
ѓ
IsVariableInitialized_22IsVariableInitialized!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_23IsVariableInitialized%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_24IsVariableInitializedconv2d_5/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel
З
IsVariableInitialized_25IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_26IsVariableInitializedbatch_normalization_5/gamma*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
°
IsVariableInitialized_27IsVariableInitializedbatch_normalization_5/beta*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_5/beta
ѓ
IsVariableInitialized_28IsVariableInitialized!batch_normalization_5/moving_mean*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_29IsVariableInitialized%batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_5/moving_variance
Л
IsVariableInitialized_30IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_31IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 
Б
IsVariableInitialized_32IsVariableInitialized
RMSprop/lr*
dtype0*
_output_shapes
: *
_class
loc:@RMSprop/lr
Г
IsVariableInitialized_33IsVariableInitializedRMSprop/rho*
_class
loc:@RMSprop/rho*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_34IsVariableInitializedRMSprop/decay* 
_class
loc:@RMSprop/decay*
dtype0*
_output_shapes
: 
С
IsVariableInitialized_35IsVariableInitializedRMSprop/iterations*%
_class
loc:@RMSprop/iterations*
dtype0	*
_output_shapes
: 
Я
IsVariableInitialized_36IsVariableInitializedtraining/RMSprop/Variable*,
_class"
 loc:@training/RMSprop/Variable*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_37IsVariableInitializedtraining/RMSprop/Variable_1*.
_class$
" loc:@training/RMSprop/Variable_1*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_38IsVariableInitializedtraining/RMSprop/Variable_2*.
_class$
" loc:@training/RMSprop/Variable_2*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_39IsVariableInitializedtraining/RMSprop/Variable_3*
dtype0*
_output_shapes
: *.
_class$
" loc:@training/RMSprop/Variable_3
£
IsVariableInitialized_40IsVariableInitializedtraining/RMSprop/Variable_4*.
_class$
" loc:@training/RMSprop/Variable_4*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_41IsVariableInitializedtraining/RMSprop/Variable_5*
dtype0*
_output_shapes
: *.
_class$
" loc:@training/RMSprop/Variable_5
£
IsVariableInitialized_42IsVariableInitializedtraining/RMSprop/Variable_6*.
_class$
" loc:@training/RMSprop/Variable_6*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_43IsVariableInitializedtraining/RMSprop/Variable_7*.
_class$
" loc:@training/RMSprop/Variable_7*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_44IsVariableInitializedtraining/RMSprop/Variable_8*.
_class$
" loc:@training/RMSprop/Variable_8*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_45IsVariableInitializedtraining/RMSprop/Variable_9*.
_class$
" loc:@training/RMSprop/Variable_9*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_46IsVariableInitializedtraining/RMSprop/Variable_10*/
_class%
#!loc:@training/RMSprop/Variable_10*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_47IsVariableInitializedtraining/RMSprop/Variable_11*/
_class%
#!loc:@training/RMSprop/Variable_11*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_48IsVariableInitializedtraining/RMSprop/Variable_12*/
_class%
#!loc:@training/RMSprop/Variable_12*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_49IsVariableInitializedtraining/RMSprop/Variable_13*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_13
•
IsVariableInitialized_50IsVariableInitializedtraining/RMSprop/Variable_14*/
_class%
#!loc:@training/RMSprop/Variable_14*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_51IsVariableInitializedtraining/RMSprop/Variable_15*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_15
•
IsVariableInitialized_52IsVariableInitializedtraining/RMSprop/Variable_16*/
_class%
#!loc:@training/RMSprop/Variable_16*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_53IsVariableInitializedtraining/RMSprop/Variable_17*/
_class%
#!loc:@training/RMSprop/Variable_17*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_54IsVariableInitializedtraining/RMSprop/Variable_18*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_18
•
IsVariableInitialized_55IsVariableInitializedtraining/RMSprop/Variable_19*/
_class%
#!loc:@training/RMSprop/Variable_19*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_56IsVariableInitializedtraining/RMSprop/Variable_20*/
_class%
#!loc:@training/RMSprop/Variable_20*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_57IsVariableInitializedtraining/RMSprop/Variable_21*/
_class%
#!loc:@training/RMSprop/Variable_21*
dtype0*
_output_shapes
: 
п
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign#^batch_normalization_1/gamma/Assign"^batch_normalization_1/beta/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign#^batch_normalization_2/gamma/Assign"^batch_normalization_2/beta/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign#^batch_normalization_3/gamma/Assign"^batch_normalization_3/beta/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign^conv2d_4/kernel/Assign^conv2d_4/bias/Assign#^batch_normalization_4/gamma/Assign"^batch_normalization_4/beta/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^conv2d_5/kernel/Assign^conv2d_5/bias/Assign#^batch_normalization_5/gamma/Assign"^batch_normalization_5/beta/Assign)^batch_normalization_5/moving_mean/Assign-^batch_normalization_5/moving_variance/Assign^conv2d_6/kernel/Assign^conv2d_6/bias/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^RMSprop/decay/Assign^RMSprop/iterations/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign#^training/RMSprop/Variable_4/Assign#^training/RMSprop/Variable_5/Assign#^training/RMSprop/Variable_6/Assign#^training/RMSprop/Variable_7/Assign#^training/RMSprop/Variable_8/Assign#^training/RMSprop/Variable_9/Assign$^training/RMSprop/Variable_10/Assign$^training/RMSprop/Variable_11/Assign$^training/RMSprop/Variable_12/Assign$^training/RMSprop/Variable_13/Assign$^training/RMSprop/Variable_14/Assign$^training/RMSprop/Variable_15/Assign$^training/RMSprop/Variable_16/Assign$^training/RMSprop/Variable_17/Assign$^training/RMSprop/Variable_18/Assign$^training/RMSprop/Variable_19/Assign$^training/RMSprop/Variable_20/Assign$^training/RMSprop/Variable_21/Assign"$нR4?     ]МѕЋ	Бn<fЊ÷AJІю
ы+џ+
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignAdd
ref"TА

value"T

output_ref"TА" 
Ttype:
2	"
use_lockingbool( 
s
	AssignSub
ref"TА

value"T

output_ref"TА" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
л
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

С
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
x
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( 
p
ResizeNearestNeighborGrad

grads"T
size
output"T"
Ttype:

2"
align_cornersbool( 
.
Rsqrt
x"T
y"T"
Ttype:

2
;
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:

2
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ц
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.7.02v1.7.0-3-g024aecf414єЙ
Е
conv2d_1_inputPlaceholder*
dtype0*1
_output_shapes
:€€€€€€€€€АА*&
shape:€€€€€€€€€АА
v
conv2d_1/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *0Њ
`
conv2d_1/random_uniform/maxConst*
valueB
 *0>*
dtype0*
_output_shapes
: 
≤
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
:*
seed2ОЏЊ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:*
T0
Й
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:*
T0
У
conv2d_1/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
»
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
[
conv2d_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
о
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0
Ш
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА
c
conv2d_1/ReluReluconv2d_1/BiasAdd*1
_output_shapes
:€€€€€€€€€АА*
T0
h
batch_normalization_1/ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
З
batch_normalization_1/gamma
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
д
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gammabatch_normalization_1/Const*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
j
batch_normalization_1/Const_1Const*
valueB*    *
dtype0*
_output_shapes
:
Ж
batch_normalization_1/beta
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
г
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/betabatch_normalization_1/Const_1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:*
use_locking(
Ы
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
j
batch_normalization_1/Const_2Const*
dtype0*
_output_shapes
:*
valueB*    
Н
!batch_normalization_1/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ш
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_meanbatch_normalization_1/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:
∞
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
j
batch_normalization_1/Const_3Const*
valueB*  А?*
dtype0*
_output_shapes
:
С
%batch_normalization_1/moving_variance
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Д
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variancebatch_normalization_1/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:
Љ
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
Й
4batch_normalization_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_1/moments/meanMeanconv2d_1/Relu4batch_normalization_1/moments/mean/reduction_indices*
T0*&
_output_shapes
:*

Tidx0*
	keep_dims(
П
*batch_normalization_1/moments/StopGradientStopGradient"batch_normalization_1/moments/mean*
T0*&
_output_shapes
:
ї
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv2d_1/Relu*batch_normalization_1/moments/StopGradient*1
_output_shapes
:€€€€€€€€€АА*
T0
Н
8batch_normalization_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_1/moments/varianceMean/batch_normalization_1/moments/SquaredDifference8batch_normalization_1/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:
Т
%batch_normalization_1/moments/SqueezeSqueeze"batch_normalization_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:
Ш
'batch_normalization_1/moments/Squeeze_1Squeeze&batch_normalization_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:
j
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_1/batchnorm/addAdd'batch_normalization_1/moments/Squeeze_1%batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes
:
x
%batch_normalization_1/batchnorm/RsqrtRsqrt#batch_normalization_1/batchnorm/add*
_output_shapes
:*
T0
Ш
#batch_normalization_1/batchnorm/mulMul%batch_normalization_1/batchnorm/Rsqrt batch_normalization_1/gamma/read*
T0*
_output_shapes
:
Ь
%batch_normalization_1/batchnorm/mul_1Mulconv2d_1/Relu#batch_normalization_1/batchnorm/mul*1
_output_shapes
:€€€€€€€€€АА*
T0
Э
%batch_normalization_1/batchnorm/mul_2Mul%batch_normalization_1/moments/Squeeze#batch_normalization_1/batchnorm/mul*
_output_shapes
:*
T0
Ч
#batch_normalization_1/batchnorm/subSubbatch_normalization_1/beta/read%batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes
:
і
%batch_normalization_1/batchnorm/add_1Add%batch_normalization_1/batchnorm/mul_1#batch_normalization_1/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
¶
+batch_normalization_1/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_1/AssignMovingAvg/subSub&batch_normalization_1/moving_mean/read%batch_normalization_1/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
г
)batch_normalization_1/AssignMovingAvg/mulMul)batch_normalization_1/AssignMovingAvg/sub+batch_normalization_1/AssignMovingAvg/decay*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
о
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
ђ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_1/AssignMovingAvg_1/subSub*batch_normalization_1/moving_variance/read'batch_normalization_1/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
н
+batch_normalization_1/AssignMovingAvg_1/mulMul+batch_normalization_1/AssignMovingAvg_1/sub-batch_normalization_1/AssignMovingAvg_1/decay*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
ъ
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
o
*batch_normalization_1/keras_learning_phasePlaceholder*
dtype0
*
_output_shapes
:*
shape:
™
!batch_normalization_1/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_1/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
Л
#batch_normalization_1/cond/Switch_1Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
Х
*batch_normalization_1/cond/batchnorm/add/yConst$^batch_normalization_1/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_1/cond/batchnorm/addAdd/batch_normalization_1/cond/batchnorm/add/Switch*batch_normalization_1/cond/batchnorm/add/y*
T0*
_output_shapes
:
о
/batch_normalization_1/cond/batchnorm/add/SwitchSwitch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
::
В
*batch_normalization_1/cond/batchnorm/RsqrtRsqrt(batch_normalization_1/cond/batchnorm/add*
T0*
_output_shapes
:
±
(batch_normalization_1/cond/batchnorm/mulMul*batch_normalization_1/cond/batchnorm/Rsqrt/batch_normalization_1/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:
Џ
/batch_normalization_1/cond/batchnorm/mul/SwitchSwitch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
 
*batch_normalization_1/cond/batchnorm/mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
й
1batch_normalization_1/cond/batchnorm/mul_1/SwitchSwitchconv2d_1/Relu"batch_normalization_1/cond/pred_id*
T0* 
_class
loc:@conv2d_1/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
≥
*batch_normalization_1/cond/batchnorm/mul_2Mul1batch_normalization_1/cond/batchnorm/mul_2/Switch(batch_normalization_1/cond/batchnorm/mul*
T0*
_output_shapes
:
и
1batch_normalization_1/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
::
±
(batch_normalization_1/cond/batchnorm/subSub/batch_normalization_1/cond/batchnorm/sub/Switch*batch_normalization_1/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Ў
/batch_normalization_1/cond/batchnorm/sub/SwitchSwitchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
√
*batch_normalization_1/cond/batchnorm/add_1Add*batch_normalization_1/cond/batchnorm/mul_1(batch_normalization_1/cond/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
√
 batch_normalization_1/cond/MergeMerge*batch_normalization_1/cond/batchnorm/add_1%batch_normalization_1/cond/Switch_1:1*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
≤
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
: *
seed2ееУ
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
: 
Й
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
: 
У
conv2d_2/kernel
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
»
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel
Ж
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: 
[
conv2d_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_2/bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
≠
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
: 
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
conv2d_2/convolutionConv2D batch_normalization_1/cond/Mergeconv2d_2/kernel/read*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ц
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@@ 
a
conv2d_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:€€€€€€€€€@@ *
T0
h
batch_normalization_2/ConstConst*
valueB *  А?*
dtype0*
_output_shapes
: 
З
batch_normalization_2/gamma
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
д
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gammabatch_normalization_2/Const*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
Ю
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
j
batch_normalization_2/Const_1Const*
valueB *    *
dtype0*
_output_shapes
: 
Ж
batch_normalization_2/beta
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
г
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/betabatch_normalization_2/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
: 
Ы
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
j
batch_normalization_2/Const_2Const*
valueB *    *
dtype0*
_output_shapes
: 
Н
!batch_normalization_2/moving_mean
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
ш
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_meanbatch_normalization_2/Const_2*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(
∞
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
j
batch_normalization_2/Const_3Const*
valueB *  А?*
dtype0*
_output_shapes
: 
С
%batch_normalization_2/moving_variance
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Д
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variancebatch_normalization_2/Const_3*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
: 
Љ
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
Й
4batch_normalization_2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_2/moments/meanMeanconv2d_2/Relu4batch_normalization_2/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
: 
П
*batch_normalization_2/moments/StopGradientStopGradient"batch_normalization_2/moments/mean*
T0*&
_output_shapes
: 
є
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv2d_2/Relu*batch_normalization_2/moments/StopGradient*/
_output_shapes
:€€€€€€€€€@@ *
T0
Н
8batch_normalization_2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_2/moments/varianceMean/batch_normalization_2/moments/SquaredDifference8batch_normalization_2/moments/variance/reduction_indices*
T0*&
_output_shapes
: *

Tidx0*
	keep_dims(
Т
%batch_normalization_2/moments/SqueezeSqueeze"batch_normalization_2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
: 
Ш
'batch_normalization_2/moments/Squeeze_1Squeeze&batch_normalization_2/moments/variance*
_output_shapes
: *
squeeze_dims
 *
T0
j
%batch_normalization_2/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_2/batchnorm/addAdd'batch_normalization_2/moments/Squeeze_1%batch_normalization_2/batchnorm/add/y*
_output_shapes
: *
T0
x
%batch_normalization_2/batchnorm/RsqrtRsqrt#batch_normalization_2/batchnorm/add*
_output_shapes
: *
T0
Ш
#batch_normalization_2/batchnorm/mulMul%batch_normalization_2/batchnorm/Rsqrt batch_normalization_2/gamma/read*
_output_shapes
: *
T0
Ъ
%batch_normalization_2/batchnorm/mul_1Mulconv2d_2/Relu#batch_normalization_2/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Э
%batch_normalization_2/batchnorm/mul_2Mul%batch_normalization_2/moments/Squeeze#batch_normalization_2/batchnorm/mul*
T0*
_output_shapes
: 
Ч
#batch_normalization_2/batchnorm/subSubbatch_normalization_2/beta/read%batch_normalization_2/batchnorm/mul_2*
T0*
_output_shapes
: 
≤
%batch_normalization_2/batchnorm/add_1Add%batch_normalization_2/batchnorm/mul_1#batch_normalization_2/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
¶
+batch_normalization_2/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_2/AssignMovingAvg/subSub&batch_normalization_2/moving_mean/read%batch_normalization_2/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
г
)batch_normalization_2/AssignMovingAvg/mulMul)batch_normalization_2/AssignMovingAvg/sub+batch_normalization_2/AssignMovingAvg/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
о
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
ђ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_2/moving_variance
ж
+batch_normalization_2/AssignMovingAvg_1/subSub*batch_normalization_2/moving_variance/read'batch_normalization_2/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
н
+batch_normalization_2/AssignMovingAvg_1/mulMul+batch_normalization_2/AssignMovingAvg_1/sub-batch_normalization_2/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
ъ
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
™
!batch_normalization_2/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_2/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
З
#batch_normalization_2/cond/Switch_1Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Х
*batch_normalization_2/cond/batchnorm/add/yConst$^batch_normalization_2/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_2/cond/batchnorm/addAdd/batch_normalization_2/cond/batchnorm/add/Switch*batch_normalization_2/cond/batchnorm/add/y*
T0*
_output_shapes
: 
о
/batch_normalization_2/cond/batchnorm/add/SwitchSwitch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
: : 
В
*batch_normalization_2/cond/batchnorm/RsqrtRsqrt(batch_normalization_2/cond/batchnorm/add*
T0*
_output_shapes
: 
±
(batch_normalization_2/cond/batchnorm/mulMul*batch_normalization_2/cond/batchnorm/Rsqrt/batch_normalization_2/cond/batchnorm/mul/Switch*
T0*
_output_shapes
: 
Џ
/batch_normalization_2/cond/batchnorm/mul/SwitchSwitch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id* 
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_2/gamma
»
*batch_normalization_2/cond/batchnorm/mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/Switch(batch_normalization_2/cond/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
е
1batch_normalization_2/cond/batchnorm/mul_1/SwitchSwitchconv2d_2/Relu"batch_normalization_2/cond/pred_id*
T0* 
_class
loc:@conv2d_2/Relu*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
≥
*batch_normalization_2/cond/batchnorm/mul_2Mul1batch_normalization_2/cond/batchnorm/mul_2/Switch(batch_normalization_2/cond/batchnorm/mul*
T0*
_output_shapes
: 
и
1batch_normalization_2/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
: : 
±
(batch_normalization_2/cond/batchnorm/subSub/batch_normalization_2/cond/batchnorm/sub/Switch*batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes
: *
T0
Ў
/batch_normalization_2/cond/batchnorm/sub/SwitchSwitchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
: : 
Ѕ
*batch_normalization_2/cond/batchnorm/add_1Add*batch_normalization_2/cond/batchnorm/mul_1(batch_normalization_2/cond/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Ѕ
 batch_normalization_2/cond/MergeMerge*batch_normalization_2/cond/batchnorm/add_1%batch_normalization_2/cond/Switch_1:1*
T0*
N*1
_output_shapes
:€€€€€€€€€@@ : 
v
conv2d_3/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
conv2d_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *  Аљ
`
conv2d_3/random_uniform/maxConst*
valueB
 *  А=*
dtype0*
_output_shapes
: 
±
%conv2d_3/random_uniform/RandomUniformRandomUniformconv2d_3/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
: @*
seed2∆ќ
}
conv2d_3/random_uniform/subSubconv2d_3/random_uniform/maxconv2d_3/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_3/random_uniform/mulMul%conv2d_3/random_uniform/RandomUniformconv2d_3/random_uniform/sub*
T0*&
_output_shapes
: @
Й
conv2d_3/random_uniformAddconv2d_3/random_uniform/mulconv2d_3/random_uniform/min*&
_output_shapes
: @*
T0
У
conv2d_3/kernel
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
»
conv2d_3/kernel/AssignAssignconv2d_3/kernelconv2d_3/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
: @
Ж
conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
: @
[
conv2d_3/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_3/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
≠
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/Const*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0* 
_class
loc:@conv2d_3/bias*
_output_shapes
:@
s
"conv2d_3/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ю
conv2d_3/convolutionConv2D batch_normalization_2/cond/Mergeconv2d_3/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€  @*
	dilations

Ц
conv2d_3/BiasAddBiasAddconv2d_3/convolutionconv2d_3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€  @
a
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€  @
h
batch_normalization_3/ConstConst*
valueB@*  А?*
dtype0*
_output_shapes
:@
З
batch_normalization_3/gamma
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
д
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gammabatch_normalization_3/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:@
Ю
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@
j
batch_normalization_3/Const_1Const*
dtype0*
_output_shapes
:@*
valueB@*    
Ж
batch_normalization_3/beta
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
г
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/betabatch_normalization_3/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:@
Ы
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:@
j
batch_normalization_3/Const_2Const*
valueB@*    *
dtype0*
_output_shapes
:@
Н
!batch_normalization_3/moving_mean
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
ш
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_meanbatch_normalization_3/Const_2*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes
:@
∞
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
j
batch_normalization_3/Const_3Const*
dtype0*
_output_shapes
:@*
valueB@*  А?
С
%batch_normalization_3/moving_variance
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Д
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variancebatch_normalization_3/Const_3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Љ
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Й
4batch_normalization_3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_3/moments/meanMeanconv2d_3/Relu4batch_normalization_3/moments/mean/reduction_indices*
T0*&
_output_shapes
:@*

Tidx0*
	keep_dims(
П
*batch_normalization_3/moments/StopGradientStopGradient"batch_normalization_3/moments/mean*
T0*&
_output_shapes
:@
є
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferenceconv2d_3/Relu*batch_normalization_3/moments/StopGradient*
T0*/
_output_shapes
:€€€€€€€€€  @
Н
8batch_normalization_3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_3/moments/varianceMean/batch_normalization_3/moments/SquaredDifference8batch_normalization_3/moments/variance/reduction_indices*
T0*&
_output_shapes
:@*

Tidx0*
	keep_dims(
Т
%batch_normalization_3/moments/SqueezeSqueeze"batch_normalization_3/moments/mean*
T0*
_output_shapes
:@*
squeeze_dims
 
Ш
'batch_normalization_3/moments/Squeeze_1Squeeze&batch_normalization_3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:@
j
%batch_normalization_3/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_3/batchnorm/addAdd'batch_normalization_3/moments/Squeeze_1%batch_normalization_3/batchnorm/add/y*
T0*
_output_shapes
:@
x
%batch_normalization_3/batchnorm/RsqrtRsqrt#batch_normalization_3/batchnorm/add*
T0*
_output_shapes
:@
Ш
#batch_normalization_3/batchnorm/mulMul%batch_normalization_3/batchnorm/Rsqrt batch_normalization_3/gamma/read*
_output_shapes
:@*
T0
Ъ
%batch_normalization_3/batchnorm/mul_1Mulconv2d_3/Relu#batch_normalization_3/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€  @
Э
%batch_normalization_3/batchnorm/mul_2Mul%batch_normalization_3/moments/Squeeze#batch_normalization_3/batchnorm/mul*
T0*
_output_shapes
:@
Ч
#batch_normalization_3/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_3/batchnorm/mul_2*
_output_shapes
:@*
T0
≤
%batch_normalization_3/batchnorm/add_1Add%batch_normalization_3/batchnorm/mul_1#batch_normalization_3/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€  @
¶
+batch_normalization_3/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_3/AssignMovingAvg/subSub&batch_normalization_3/moving_mean/read%batch_normalization_3/moments/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
г
)batch_normalization_3/AssignMovingAvg/mulMul)batch_normalization_3/AssignMovingAvg/sub+batch_normalization_3/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
о
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
ђ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_3/AssignMovingAvg_1/subSub*batch_normalization_3/moving_variance/read'batch_normalization_3/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@
н
+batch_normalization_3/AssignMovingAvg_1/mulMul+batch_normalization_3/AssignMovingAvg_1/sub-batch_normalization_3/AssignMovingAvg_1/decay*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
ъ
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@
™
!batch_normalization_3/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_3/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes
:
З
#batch_normalization_3/cond/Switch_1Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
Х
*batch_normalization_3/cond/batchnorm/add/yConst$^batch_normalization_3/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_3/cond/batchnorm/addAdd/batch_normalization_3/cond/batchnorm/add/Switch*batch_normalization_3/cond/batchnorm/add/y*
_output_shapes
:@*
T0
о
/batch_normalization_3/cond/batchnorm/add/SwitchSwitch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance* 
_output_shapes
:@:@
В
*batch_normalization_3/cond/batchnorm/RsqrtRsqrt(batch_normalization_3/cond/batchnorm/add*
T0*
_output_shapes
:@
±
(batch_normalization_3/cond/batchnorm/mulMul*batch_normalization_3/cond/batchnorm/Rsqrt/batch_normalization_3/cond/batchnorm/mul/Switch*
_output_shapes
:@*
T0
Џ
/batch_normalization_3/cond/batchnorm/mul/SwitchSwitch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:@:@
»
*batch_normalization_3/cond/batchnorm/mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€  @
е
1batch_normalization_3/cond/batchnorm/mul_1/SwitchSwitchconv2d_3/Relu"batch_normalization_3/cond/pred_id*
T0* 
_class
loc:@conv2d_3/Relu*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
≥
*batch_normalization_3/cond/batchnorm/mul_2Mul1batch_normalization_3/cond/batchnorm/mul_2/Switch(batch_normalization_3/cond/batchnorm/mul*
T0*
_output_shapes
:@
и
1batch_normalization_3/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean* 
_output_shapes
:@:@
±
(batch_normalization_3/cond/batchnorm/subSub/batch_normalization_3/cond/batchnorm/sub/Switch*batch_normalization_3/cond/batchnorm/mul_2*
T0*
_output_shapes
:@
Ў
/batch_normalization_3/cond/batchnorm/sub/SwitchSwitchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
:@:@
Ѕ
*batch_normalization_3/cond/batchnorm/add_1Add*batch_normalization_3/cond/batchnorm/mul_1(batch_normalization_3/cond/batchnorm/sub*/
_output_shapes
:€€€€€€€€€  @*
T0
Ѕ
 batch_normalization_3/cond/MergeMerge*batch_normalization_3/cond/batchnorm/add_1%batch_normalization_3/cond/Switch_1:1*
T0*
N*1
_output_shapes
:€€€€€€€€€  @: 
u
up_sampling2d_1/ShapeShape batch_normalization_3/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ќ
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
T0*
_output_shapes
:
ƒ
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_3/cond/Mergeup_sampling2d_1/mul*/
_output_shapes
:€€€€€€€€€@@@*
align_corners( *
T0
v
conv2d_4/random_uniform/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
`
conv2d_4/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
`
conv2d_4/random_uniform/maxConst*
valueB
 *  А=*
dtype0*
_output_shapes
: 
±
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
:@ *
seed2•Ь\
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*&
_output_shapes
:@ *
T0
Й
conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*&
_output_shapes
:@ *
T0
У
conv2d_4/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@ *
	container *
shape:@ 
»
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:@ 
Ж
conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@ 
[
conv2d_4/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
y
conv2d_4/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
≠
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
: *
use_locking(
t
conv2d_4/bias/readIdentityconv2d_4/bias*
T0* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Г
conv2d_4/convolutionConv2D%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ 
Ц
conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@@ 
a
conv2d_4/ReluReluconv2d_4/BiasAdd*/
_output_shapes
:€€€€€€€€€@@ *
T0
h
batch_normalization_4/ConstConst*
dtype0*
_output_shapes
: *
valueB *  А?
З
batch_normalization_4/gamma
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
д
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gammabatch_normalization_4/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
: 
Ю
 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_4/gamma
j
batch_normalization_4/Const_1Const*
valueB *    *
dtype0*
_output_shapes
: 
Ж
batch_normalization_4/beta
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
г
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/betabatch_normalization_4/Const_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
: *
use_locking(
Ы
batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_4/beta
j
batch_normalization_4/Const_2Const*
valueB *    *
dtype0*
_output_shapes
: 
Н
!batch_normalization_4/moving_mean
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ш
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_meanbatch_normalization_4/Const_2*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(
∞
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
j
batch_normalization_4/Const_3Const*
valueB *  А?*
dtype0*
_output_shapes
: 
С
%batch_normalization_4/moving_variance
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Д
,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variancebatch_normalization_4/Const_3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
Љ
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
Й
4batch_normalization_4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
љ
"batch_normalization_4/moments/meanMeanconv2d_4/Relu4batch_normalization_4/moments/mean/reduction_indices*&
_output_shapes
: *

Tidx0*
	keep_dims(*
T0
П
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
T0*&
_output_shapes
: 
є
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceconv2d_4/Relu*batch_normalization_4/moments/StopGradient*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Н
8batch_normalization_4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
T0*&
_output_shapes
: *

Tidx0*
	keep_dims(
Т
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
: 
Ш
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
_output_shapes
: *
squeeze_dims
 *
T0
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_4/batchnorm/addAdd'batch_normalization_4/moments/Squeeze_1%batch_normalization_4/batchnorm/add/y*
_output_shapes
: *
T0
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
T0*
_output_shapes
: 
Ш
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_4/gamma/read*
_output_shapes
: *
T0
Ъ
%batch_normalization_4/batchnorm/mul_1Mulconv2d_4/Relu#batch_normalization_4/batchnorm/mul*/
_output_shapes
:€€€€€€€€€@@ *
T0
Э
%batch_normalization_4/batchnorm/mul_2Mul%batch_normalization_4/moments/Squeeze#batch_normalization_4/batchnorm/mul*
_output_shapes
: *
T0
Ч
#batch_normalization_4/batchnorm/subSubbatch_normalization_4/beta/read%batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes
: 
≤
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
¶
+batch_normalization_4/AssignMovingAvg/decayConst*
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Џ
)batch_normalization_4/AssignMovingAvg/subSub&batch_normalization_4/moving_mean/read%batch_normalization_4/moments/Squeeze*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
г
)batch_normalization_4/AssignMovingAvg/mulMul)batch_normalization_4/AssignMovingAvg/sub+batch_normalization_4/AssignMovingAvg/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
о
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
ђ
-batch_normalization_4/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_4/AssignMovingAvg_1/subSub*batch_normalization_4/moving_variance/read'batch_normalization_4/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
н
+batch_normalization_4/AssignMovingAvg_1/mulMul+batch_normalization_4/AssignMovingAvg_1/sub-batch_normalization_4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
ъ
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
™
!batch_normalization_4/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
}
"batch_normalization_4/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
_output_shapes
:*
T0

З
#batch_normalization_4/cond/Switch_1Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Х
*batch_normalization_4/cond/batchnorm/add/yConst$^batch_normalization_4/cond/switch_f*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
±
(batch_normalization_4/cond/batchnorm/addAdd/batch_normalization_4/cond/batchnorm/add/Switch*batch_normalization_4/cond/batchnorm/add/y*
T0*
_output_shapes
: 
о
/batch_normalization_4/cond/batchnorm/add/SwitchSwitch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance* 
_output_shapes
: : 
В
*batch_normalization_4/cond/batchnorm/RsqrtRsqrt(batch_normalization_4/cond/batchnorm/add*
_output_shapes
: *
T0
±
(batch_normalization_4/cond/batchnorm/mulMul*batch_normalization_4/cond/batchnorm/Rsqrt/batch_normalization_4/cond/batchnorm/mul/Switch*
T0*
_output_shapes
: 
Џ
/batch_normalization_4/cond/batchnorm/mul/SwitchSwitch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id* 
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_4/gamma
»
*batch_normalization_4/cond/batchnorm/mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*/
_output_shapes
:€€€€€€€€€@@ 
е
1batch_normalization_4/cond/batchnorm/mul_1/SwitchSwitchconv2d_4/Relu"batch_normalization_4/cond/pred_id*
T0* 
_class
loc:@conv2d_4/Relu*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
≥
*batch_normalization_4/cond/batchnorm/mul_2Mul1batch_normalization_4/cond/batchnorm/mul_2/Switch(batch_normalization_4/cond/batchnorm/mul*
T0*
_output_shapes
: 
и
1batch_normalization_4/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean* 
_output_shapes
: : 
±
(batch_normalization_4/cond/batchnorm/subSub/batch_normalization_4/cond/batchnorm/sub/Switch*batch_normalization_4/cond/batchnorm/mul_2*
T0*
_output_shapes
: 
Ў
/batch_normalization_4/cond/batchnorm/sub/SwitchSwitchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
: : 
Ѕ
*batch_normalization_4/cond/batchnorm/add_1Add*batch_normalization_4/cond/batchnorm/mul_1(batch_normalization_4/cond/batchnorm/sub*
T0*/
_output_shapes
:€€€€€€€€€@@ 
Ѕ
 batch_normalization_4/cond/MergeMerge*batch_normalization_4/cond/batchnorm/add_1%batch_normalization_4/cond/Switch_1:1*
T0*
N*1
_output_shapes
:€€€€€€€€€@@ : 
u
up_sampling2d_2/ShapeShape batch_normalization_4/cond/Merge*
_output_shapes
:*
T0*
out_type0
m
#up_sampling2d_2/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape#up_sampling2d_2/strided_slice/stack%up_sampling2d_2/strided_slice/stack_1%up_sampling2d_2/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
f
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_2/mulMulup_sampling2d_2/strided_sliceup_sampling2d_2/Const*
T0*
_output_shapes
:
∆
%up_sampling2d_2/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_4/cond/Mergeup_sampling2d_2/mul*
T0*1
_output_shapes
:€€€€€€€€€АА *
align_corners( 
v
conv2d_5/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_5/random_uniform/minConst*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
≤
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: *
seed2ЩЃИ*
seed±€е)
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*
T0*&
_output_shapes
: 
Й
conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*&
_output_shapes
: 
У
conv2d_5/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
»
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
: 
Ж
conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
: 
[
conv2d_5/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_5/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias
t
conv2d_5/bias/readIdentityconv2d_5/bias*
T0* 
_class
loc:@conv2d_5/bias*
_output_shapes
:
s
"conv2d_5/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Е
conv2d_5/convolutionConv2D%up_sampling2d_2/ResizeNearestNeighborconv2d_5/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА
Ш
conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА*
T0
c
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*1
_output_shapes
:€€€€€€€€€АА
h
batch_normalization_5/ConstConst*
dtype0*
_output_shapes
:*
valueB*  А?
З
batch_normalization_5/gamma
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
д
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gammabatch_normalization_5/Const*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:
Ю
 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
j
batch_normalization_5/Const_1Const*
valueB*    *
dtype0*
_output_shapes
:
Ж
batch_normalization_5/beta
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
г
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/betabatch_normalization_5/Const_1*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes
:
Ы
batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_5/beta
j
batch_normalization_5/Const_2Const*
dtype0*
_output_shapes
:*
valueB*    
Н
!batch_normalization_5/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ш
(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_meanbatch_normalization_5/Const_2*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(
∞
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
j
batch_normalization_5/Const_3Const*
dtype0*
_output_shapes
:*
valueB*  А?
С
%batch_normalization_5/moving_variance
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Д
,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variancebatch_normalization_5/Const_3*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(
Љ
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
Й
4batch_normalization_5/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
љ
"batch_normalization_5/moments/meanMeanconv2d_5/Relu4batch_normalization_5/moments/mean/reduction_indices*&
_output_shapes
:*

Tidx0*
	keep_dims(*
T0
П
*batch_normalization_5/moments/StopGradientStopGradient"batch_normalization_5/moments/mean*
T0*&
_output_shapes
:
ї
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferenceconv2d_5/Relu*batch_normalization_5/moments/StopGradient*
T0*1
_output_shapes
:€€€€€€€€€АА
Н
8batch_normalization_5/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
з
&batch_normalization_5/moments/varianceMean/batch_normalization_5/moments/SquaredDifference8batch_normalization_5/moments/variance/reduction_indices*
T0*&
_output_shapes
:*

Tidx0*
	keep_dims(
Т
%batch_normalization_5/moments/SqueezeSqueeze"batch_normalization_5/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:
Ш
'batch_normalization_5/moments/Squeeze_1Squeeze&batch_normalization_5/moments/variance*
T0*
_output_shapes
:*
squeeze_dims
 
j
%batch_normalization_5/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Я
#batch_normalization_5/batchnorm/addAdd'batch_normalization_5/moments/Squeeze_1%batch_normalization_5/batchnorm/add/y*
_output_shapes
:*
T0
x
%batch_normalization_5/batchnorm/RsqrtRsqrt#batch_normalization_5/batchnorm/add*
_output_shapes
:*
T0
Ш
#batch_normalization_5/batchnorm/mulMul%batch_normalization_5/batchnorm/Rsqrt batch_normalization_5/gamma/read*
T0*
_output_shapes
:
Ь
%batch_normalization_5/batchnorm/mul_1Mulconv2d_5/Relu#batch_normalization_5/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
Э
%batch_normalization_5/batchnorm/mul_2Mul%batch_normalization_5/moments/Squeeze#batch_normalization_5/batchnorm/mul*
T0*
_output_shapes
:
Ч
#batch_normalization_5/batchnorm/subSubbatch_normalization_5/beta/read%batch_normalization_5/batchnorm/mul_2*
_output_shapes
:*
T0
і
%batch_normalization_5/batchnorm/add_1Add%batch_normalization_5/batchnorm/mul_1#batch_normalization_5/batchnorm/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
¶
+batch_normalization_5/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<*4
_class*
(&loc:@batch_normalization_5/moving_mean
Џ
)batch_normalization_5/AssignMovingAvg/subSub&batch_normalization_5/moving_mean/read%batch_normalization_5/moments/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
г
)batch_normalization_5/AssignMovingAvg/mulMul)batch_normalization_5/AssignMovingAvg/sub+batch_normalization_5/AssignMovingAvg/decay*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:
о
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_5/AssignMovingAvg/mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
ђ
-batch_normalization_5/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: 
ж
+batch_normalization_5/AssignMovingAvg_1/subSub*batch_normalization_5/moving_variance/read'batch_normalization_5/moments/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:
н
+batch_normalization_5/AssignMovingAvg_1/mulMul+batch_normalization_5/AssignMovingAvg_1/sub-batch_normalization_5/AssignMovingAvg_1/decay*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:
ъ
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_5/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:
™
!batch_normalization_5/cond/SwitchSwitch*batch_normalization_1/keras_learning_phase*batch_normalization_1/keras_learning_phase*
T0
*
_output_shapes

::
w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
_output_shapes
:*
T0

}
"batch_normalization_5/cond/pred_idIdentity*batch_normalization_1/keras_learning_phase*
_output_shapes
:*
T0

Л
#batch_normalization_5/cond/Switch_1Switch%batch_normalization_5/batchnorm/add_1"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Х
*batch_normalization_5/cond/batchnorm/add/yConst$^batch_normalization_5/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *oГ:
±
(batch_normalization_5/cond/batchnorm/addAdd/batch_normalization_5/cond/batchnorm/add/Switch*batch_normalization_5/cond/batchnorm/add/y*
T0*
_output_shapes
:
о
/batch_normalization_5/cond/batchnorm/add/SwitchSwitch*batch_normalization_5/moving_variance/read"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance* 
_output_shapes
::
В
*batch_normalization_5/cond/batchnorm/RsqrtRsqrt(batch_normalization_5/cond/batchnorm/add*
T0*
_output_shapes
:
±
(batch_normalization_5/cond/batchnorm/mulMul*batch_normalization_5/cond/batchnorm/Rsqrt/batch_normalization_5/cond/batchnorm/mul/Switch*
T0*
_output_shapes
:
Џ
/batch_normalization_5/cond/batchnorm/mul/SwitchSwitch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma* 
_output_shapes
::
 
*batch_normalization_5/cond/batchnorm/mul_1Mul1batch_normalization_5/cond/batchnorm/mul_1/Switch(batch_normalization_5/cond/batchnorm/mul*
T0*1
_output_shapes
:€€€€€€€€€АА
й
1batch_normalization_5/cond/batchnorm/mul_1/SwitchSwitchconv2d_5/Relu"batch_normalization_5/cond/pred_id*
T0* 
_class
loc:@conv2d_5/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
≥
*batch_normalization_5/cond/batchnorm/mul_2Mul1batch_normalization_5/cond/batchnorm/mul_2/Switch(batch_normalization_5/cond/batchnorm/mul*
T0*
_output_shapes
:
и
1batch_normalization_5/cond/batchnorm/mul_2/SwitchSwitch&batch_normalization_5/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean* 
_output_shapes
::
±
(batch_normalization_5/cond/batchnorm/subSub/batch_normalization_5/cond/batchnorm/sub/Switch*batch_normalization_5/cond/batchnorm/mul_2*
T0*
_output_shapes
:
Ў
/batch_normalization_5/cond/batchnorm/sub/SwitchSwitchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_5/beta
√
*batch_normalization_5/cond/batchnorm/add_1Add*batch_normalization_5/cond/batchnorm/mul_1(batch_normalization_5/cond/batchnorm/sub*1
_output_shapes
:€€€€€€€€€АА*
T0
√
 batch_normalization_5/cond/MergeMerge*batch_normalization_5/cond/batchnorm/add_1%batch_normalization_5/cond/Switch_1:1*
T0*
N*3
_output_shapes!
:€€€€€€€€€АА: 
u
up_sampling2d_3/ShapeShape batch_normalization_5/cond/Merge*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape#up_sampling2d_3/strided_slice/stack%up_sampling2d_3/strided_slice/stack_1%up_sampling2d_3/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*
shrink_axis_mask 
f
up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_3/mulMulup_sampling2d_3/strided_sliceup_sampling2d_3/Const*
T0*
_output_shapes
:
∆
%up_sampling2d_3/ResizeNearestNeighborResizeNearestNeighbor batch_normalization_5/cond/Mergeup_sampling2d_3/mul*
align_corners( *
T0*1
_output_shapes
:€€€€€€€€€АА
v
conv2d_6/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_6/random_uniform/minConst*
valueB
 *:ЌЊ*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
valueB
 *:Ќ>*
dtype0*
_output_shapes
: 
≤
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
:*
seed2Ы€Ю
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*
T0*&
_output_shapes
:
Й
conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*&
_output_shapes
:*
T0
У
conv2d_6/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
»
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:
Ж
conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
[
conv2d_6/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_6/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
≠
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Е
conv2d_6/convolutionConv2D%up_sampling2d_3/ResizeNearestNeighborconv2d_6/kernel/read*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ш
conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*
T0*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА
c
conv2d_6/TanhTanhconv2d_6/BiasAdd*
T0*1
_output_shapes
:€€€€€€€€€АА
]
RMSprop/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:
n

RMSprop/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
™
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
use_locking(*
T0*
_class
loc:@RMSprop/lr*
validate_shape(*
_output_shapes
: 
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
RMSprop/rho
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ѓ
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@RMSprop/rho
j
RMSprop/rho/readIdentityRMSprop/rho*
_output_shapes
: *
T0*
_class
loc:@RMSprop/rho
`
RMSprop/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
q
RMSprop/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ґ
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
use_locking(*
T0* 
_class
loc:@RMSprop/decay*
validate_shape(*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
 
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*%
_class
loc:@RMSprop/iterations

RMSprop/iterations/readIdentityRMSprop/iterations*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
Є
conv2d_6_targetPlaceholder*?
shape6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
r
conv2d_6_sample_weightsPlaceholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
]
loss/conv2d_6_loss/ConstConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
]
loss/conv2d_6_loss/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
loss/conv2d_6_loss/subSubloss/conv2d_6_loss/sub/xloss/conv2d_6_loss/Const*
T0*
_output_shapes
: 
Ц
(loss/conv2d_6_loss/clip_by_value/MinimumMinimumconv2d_6/Tanhloss/conv2d_6_loss/sub*
T0*1
_output_shapes
:€€€€€€€€€АА
Ђ
 loss/conv2d_6_loss/clip_by_valueMaximum(loss/conv2d_6_loss/clip_by_value/Minimumloss/conv2d_6_loss/Const*
T0*1
_output_shapes
:€€€€€€€€€АА
_
loss/conv2d_6_loss/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
loss/conv2d_6_loss/sub_1Subloss/conv2d_6_loss/sub_1/x loss/conv2d_6_loss/clip_by_value*
T0*1
_output_shapes
:€€€€€€€€€АА
Э
loss/conv2d_6_loss/truedivRealDiv loss/conv2d_6_loss/clip_by_valueloss/conv2d_6_loss/sub_1*
T0*1
_output_shapes
:€€€€€€€€€АА
u
loss/conv2d_6_loss/LogLogloss/conv2d_6_loss/truediv*
T0*1
_output_shapes
:€€€€€€€€€АА
М
+loss/conv2d_6_loss/logistic_loss/zeros_like	ZerosLikeloss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА*
T0
Њ
-loss/conv2d_6_loss/logistic_loss/GreaterEqualGreaterEqualloss/conv2d_6_loss/Log+loss/conv2d_6_loss/logistic_loss/zeros_like*
T0*1
_output_shapes
:€€€€€€€€€АА
б
'loss/conv2d_6_loss/logistic_loss/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualloss/conv2d_6_loss/Log+loss/conv2d_6_loss/logistic_loss/zeros_like*
T0*1
_output_shapes
:€€€€€€€€€АА

$loss/conv2d_6_loss/logistic_loss/NegNegloss/conv2d_6_loss/Log*
T0*1
_output_shapes
:€€€€€€€€€АА
№
)loss/conv2d_6_loss/logistic_loss/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqual$loss/conv2d_6_loss/logistic_loss/Negloss/conv2d_6_loss/Log*
T0*1
_output_shapes
:€€€€€€€€€АА
Р
$loss/conv2d_6_loss/logistic_loss/mulMulloss/conv2d_6_loss/Logconv2d_6_target*
T0*1
_output_shapes
:€€€€€€€€€АА
ґ
$loss/conv2d_6_loss/logistic_loss/subSub'loss/conv2d_6_loss/logistic_loss/Select$loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА*
T0
Т
$loss/conv2d_6_loss/logistic_loss/ExpExp)loss/conv2d_6_loss/logistic_loss/Select_1*
T0*1
_output_shapes
:€€€€€€€€€АА
С
&loss/conv2d_6_loss/logistic_loss/Log1pLog1p$loss/conv2d_6_loss/logistic_loss/Exp*
T0*1
_output_shapes
:€€€€€€€€€АА
±
 loss/conv2d_6_loss/logistic_lossAdd$loss/conv2d_6_loss/logistic_loss/sub&loss/conv2d_6_loss/logistic_loss/Log1p*1
_output_shapes
:€€€€€€€€€АА*
T0
t
)loss/conv2d_6_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ѕ
loss/conv2d_6_loss/MeanMean loss/conv2d_6_loss/logistic_loss)loss/conv2d_6_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*-
_output_shapes
:€€€€€€€€€АА
|
+loss/conv2d_6_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
≤
loss/conv2d_6_loss/Mean_1Meanloss/conv2d_6_loss/Mean+loss/conv2d_6_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 

loss/conv2d_6_loss/mulMulloss/conv2d_6_loss/Mean_1conv2d_6_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
b
loss/conv2d_6_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
loss/conv2d_6_loss/NotEqualNotEqualconv2d_6_sample_weightsloss/conv2d_6_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
y
loss/conv2d_6_loss/CastCastloss/conv2d_6_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
d
loss/conv2d_6_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
loss/conv2d_6_loss/Mean_2Meanloss/conv2d_6_loss/Castloss/conv2d_6_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
И
loss/conv2d_6_loss/truediv_1RealDivloss/conv2d_6_loss/mulloss/conv2d_6_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
d
loss/conv2d_6_loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
Щ
loss/conv2d_6_loss/Mean_3Meanloss/conv2d_6_loss/truediv_1loss/conv2d_6_loss/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/conv2d_6_loss/Mean_3*
T0*
_output_shapes
: 
А
 training/RMSprop/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
Ж
$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  А?*
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
њ
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: 
≠
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/conv2d_6_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul
†
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
њ
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
¶
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
:
…
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ShapeShapeloss/conv2d_6_loss/truediv_1*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
:
Є
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/TileTileAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Reshape?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape*#
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
Ћ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_1Shapeloss/conv2d_6_loss/truediv_1*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
:
≤
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_2Const*
valueB *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
dtype0*
_output_shapes
: 
Ј
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
ґ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ProdProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_1?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
Ї
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Prod_1ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Shape_2Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Const_1*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
≥
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum/yConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
dtype0*
_output_shapes
: 
Ґ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/MaximumMaximum@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Prod_1Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum/y*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3*
_output_shapes
: 
†
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/floordivFloorDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
и
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/CastCastBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
®
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivRealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Tile>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_3
…
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ShapeShapeloss/conv2d_6_loss/mul*
_output_shapes
:*
T0*
out_type0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
Є
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB */
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
г
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ShapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDivRealDivAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivloss/conv2d_6_loss/Mean_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
“
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/SumSumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDivRtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
¬
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/ReshapeReshape@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/SumBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape*
T0*
Tshape0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Њ
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/NegNegloss/conv2d_6_loss/mul*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Н
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_1RealDiv@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Negloss/conv2d_6_loss/Mean_2*#
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1
У
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_2RealDivFtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_1loss/conv2d_6_loss/Mean_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
±
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/mulMulAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_3_grad/truedivFtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/RealDiv_2*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
“
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Sum_1Sum@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/mulTtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshape_1ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Sum_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Shape_1*
T0*
Tshape0*/
_class%
#!loc:@loss/conv2d_6_loss/truediv_1*
_output_shapes
: 
ј
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ShapeShapeloss/conv2d_6_loss/Mean_1*
T0*
out_type0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:
ј
>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1Shapeconv2d_6_sample_weights*
T0*
out_type0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:
Ћ
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
щ
:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/MulMulDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshapeconv2d_6_sample_weights*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
ґ
:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/SumSum:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/MulLtraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/conv2d_6_loss/mul
™
>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ReshapeReshape:training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape*
T0*
Tshape0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
э
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Mul_1Mulloss/conv2d_6_loss/Mean_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_1_grad/Reshape*#
_output_shapes
:€€€€€€€€€*
T0*)
_class
loc:@loss/conv2d_6_loss/mul
Љ
<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum_1Sum<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Mul_1Ntraining/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/conv2d_6_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
∞
@training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Reshape_1Reshape<training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Sum_1>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/conv2d_6_loss/mul*#
_output_shapes
:€€€€€€€€€
ƒ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ShapeShapeloss/conv2d_6_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
Ѓ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
Д
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/addAdd+loss/conv2d_6_loss/Mean_1/reduction_indices>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
Ы
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/modFloorMod=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/add>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Size*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_1Const*
valueB:*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
:
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/startConst*
value	B : *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/deltaConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
м
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/rangeRangeEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/start>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/SizeEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range/delta*

Tidx0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
і
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill/valueConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
і
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/FillFillAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_1Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
Њ
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitchDynamicStitch?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/range=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/mod?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Fill*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
N*#
_output_shapes
:€€€€€€€€€
≥
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
ґ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/MaximumMaximumGtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitchCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
•
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordivFloorDiv?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ShapeAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
і
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ReshapeReshape>training/RMSprop/gradients/loss/conv2d_6_loss/mul_grad/ReshapeGtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
’
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/TileTileAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
∆
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_2Shapeloss/conv2d_6_loss/Mean*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
:
»
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_3Shapeloss/conv2d_6_loss/Mean_1*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1
Ј
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ConstConst*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
:
ґ
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ProdProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_2?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
є
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const_1Const*
valueB: *,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
:
Ї
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Prod_1ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Shape_3Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
µ
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
dtype0*
_output_shapes
: 
¶
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1Maximum@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Prod_1Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1/y*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
§
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv_1FloorDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/ProdCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Maximum_1*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: 
к
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/CastCastDtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/floordiv_1*

SrcT0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*
_output_shapes
: *

DstT0
≤
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/truedivRealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Tile>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/Cast*
T0*,
_class"
 loc:@loss/conv2d_6_loss/Mean_1*-
_output_shapes
:€€€€€€€€€АА
…
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ShapeShape loss/conv2d_6_loss/logistic_loss*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean
™
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/SizeConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
ш
;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/addAdd)loss/conv2d_6_loss/Mean/reduction_indices<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Size*
_output_shapes
: *
T0**
_class 
loc:@loss/conv2d_6_loss/Mean
П
;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/modFloorMod;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/add<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Size*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB **
_class 
loc:@loss/conv2d_6_loss/Mean
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/startConst*
value	B : **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/deltaConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
в
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/rangeRangeCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/start<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/SizeCtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range/delta*

Tidx0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
∞
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill/valueConst*
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
: 
®
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/FillFill?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_1Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill/value*
T0*

index_type0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
≤
Etraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitchDynamicStitch=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/range;training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/mod=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Fill*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
N*#
_output_shapes
:€€€€€€€€€
ѓ
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/MaximumMaximumEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitchAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum/y*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*#
_output_shapes
:€€€€€€€€€
Э
@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordivFloorDiv=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum*
_output_shapes
:*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean
±
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ReshapeReshapeAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_1_grad/truedivEtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/DynamicStitch*
T0*
Tshape0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
Џ
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/TileTile?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Reshape@training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv*

Tmultiples0*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ћ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_2Shape loss/conv2d_6_loss/logistic_loss*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
:
¬
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_3Shapeloss/conv2d_6_loss/Mean*
_output_shapes
:*
T0*
out_type0**
_class 
loc:@loss/conv2d_6_loss/Mean
≥
=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ConstConst*
valueB: **
_class 
loc:@loss/conv2d_6_loss/Mean*
dtype0*
_output_shapes
:
Ѓ
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ProdProd?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_2=training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
µ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: **
_class 
loc:@loss/conv2d_6_loss/Mean
≤
>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Prod_1Prod?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Shape_3?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
±
Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :**
_class 
loc:@loss/conv2d_6_loss/Mean
Ю
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1Maximum>training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Prod_1Ctraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1/y*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
Ь
Btraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv_1FloorDiv<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/ProdAtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Maximum_1*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: 
д
<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/CastCastBtraining/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/floordiv_1*

SrcT0**
_class 
loc:@loss/conv2d_6_loss/Mean*
_output_shapes
: *

DstT0
Ѓ
?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivRealDiv<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Tile<training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/Cast*
T0**
_class 
loc:@loss/conv2d_6_loss/Mean*1
_output_shapes
:€€€€€€€€€АА
я
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ShapeShape$loss/conv2d_6_loss/logistic_loss/sub*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*
_output_shapes
:
г
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1Shape&loss/conv2d_6_loss/logistic_loss/Log1p*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*
_output_shapes
:
у
Vtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss
ў
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/SumSum?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivVtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss
а
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ReshapeReshapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/SumFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*1
_output_shapes
:€€€€€€€€€АА
Ё
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Sum_1Sum?training/RMSprop/gradients/loss/conv2d_6_loss/Mean_grad/truedivXtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*
_output_shapes
:
ж
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1ReshapeFtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Sum_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Shape_1*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/logistic_loss*1
_output_shapes
:€€€€€€€€€АА
к
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ShapeShape'loss/conv2d_6_loss/logistic_loss/Select*
_output_shapes
:*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub
й
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1Shape$loss/conv2d_6_loss/logistic_loss/mul*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
Г
Ztraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ShapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/SumSumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/ReshapeZtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
р
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ReshapeReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*1
_output_shapes
:€€€€€€€€€АА
т
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Sum_1SumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape\training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/NegNegJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Sum_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*
_output_shapes
:
ф
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1ReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/NegLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/sub*1
_output_shapes
:€€€€€€€€€АА
Щ
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add/xConstK^training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1*
valueB
 *  А?*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p*
dtype0*
_output_shapes
: 
Љ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/addAddLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add/x$loss/conv2d_6_loss/logistic_loss/Exp*1
_output_shapes
:€€€€€€€€€АА*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p
Ґ
Qtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/Reciprocal
ReciprocalJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/add*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p*1
_output_shapes
:€€€€€€€€€АА
з
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/mulMulJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss_grad/Reshape_1Qtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/Reciprocal*
T0*9
_class/
-+loc:@loss/conv2d_6_loss/logistic_loss/Log1p*1
_output_shapes
:€€€€€€€€€АА
п
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_like	ZerosLikeloss/conv2d_6_loss/Log*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*1
_output_shapes
:€€€€€€€€€АА
°
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/ReshapeRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_like*1
_output_shapes
:€€€€€€€€€АА*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select
£
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqualRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/zeros_likeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*1
_output_shapes
:€€€€€€€€€АА
ў
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ShapeShapeloss/conv2d_6_loss/Log*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*
_output_shapes
:
‘
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1Shapeconv2d_6_target*
_output_shapes
:*
T0*
out_type0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
Г
Ztraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ShapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/MulMulNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1conv2d_6_target*1
_output_shapes
:€€€€€€€€€АА*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
о
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/SumSumHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/MulZtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul
р
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ReshapeReshapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА
Ѓ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Mul_1Mulloss/conv2d_6_loss/LogNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/sub_grad/Reshape_1*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*1
_output_shapes
:€€€€€€€€€АА
ф
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Sum_1SumJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Mul_1\training/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*
_output_shapes
:
П
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Reshape_1ReshapeJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Sum_1Ltraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/mul*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ґ
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mulMulJtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Log1p_grad/mul$loss/conv2d_6_loss/logistic_loss/Exp*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/Exp*1
_output_shapes
:€€€€€€€€€АА
Б
Ttraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_like	ZerosLike$loss/conv2d_6_loss/logistic_loss/Neg*1
_output_shapes
:€€€€€€€€€АА*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1
£
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/SelectSelect-loss/conv2d_6_loss/logistic_loss/GreaterEqualHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mulTtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_like*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1*1
_output_shapes
:€€€€€€€€€АА
•
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select_1Select-loss/conv2d_6_loss/logistic_loss/GreaterEqualTtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/zeros_likeHtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Exp_grad/mul*1
_output_shapes
:€€€€€€€€€АА*
T0*<
_class2
0.loc:@loss/conv2d_6_loss/logistic_loss/Select_1
Ц
Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Neg_grad/NegNegPtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select*
T0*7
_class-
+)loc:@loss/conv2d_6_loss/logistic_loss/Neg*1
_output_shapes
:€€€€€€€€€АА
д
training/RMSprop/gradients/AddNAddNNtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_grad/SelectLtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/mul_grad/ReshapeRtraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Select_1_grad/Select_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/logistic_loss/Neg_grad/Neg*
T0*:
_class0
.,loc:@loss/conv2d_6_loss/logistic_loss/Select*
N*1
_output_shapes
:€€€€€€€€€АА
ф
Atraining/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/Reciprocal
Reciprocalloss/conv2d_6_loss/truediv ^training/RMSprop/gradients/AddN*
T0*)
_class
loc:@loss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА
М
:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulMultraining/RMSprop/gradients/AddNAtraining/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/Reciprocal*
T0*)
_class
loc:@loss/conv2d_6_loss/Log*1
_output_shapes
:€€€€€€€€€АА
ѕ
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ShapeShape loss/conv2d_6_loss/clip_by_value*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
…
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1Shapeloss/conv2d_6_loss/sub_1*
T0*
out_type0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*
_output_shapes
:
џ
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ShapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDivRealDiv:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulloss/conv2d_6_loss/sub_1*1
_output_shapes
:€€€€€€€€€АА*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
 
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/SumSumBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDivPtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
»
Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ReshapeReshape>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
“
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/NegNeg loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
Ф
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_1RealDiv>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Negloss/conv2d_6_loss/sub_1*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
Ъ
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_2RealDivDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_1loss/conv2d_6_loss/sub_1*1
_output_shapes
:€€€€€€€€€АА*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
≤
>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/mulMul:training/RMSprop/gradients/loss/conv2d_6_loss/Log_grad/mulDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/RealDiv_2*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
 
@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum_1Sum>training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/mulRtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
ќ
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Reshape@training/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Sum_1Btraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@loss/conv2d_6_loss/truediv*1
_output_shapes
:€€€€€€€€€АА
Ѓ
>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/ShapeConst*
valueB *+
_class!
loc:@loss/conv2d_6_loss/sub_1*
dtype0*
_output_shapes
: 
Ќ
@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1Shape loss/conv2d_6_loss/clip_by_value*
T0*
out_type0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:
”
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∆
<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/SumSumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Ntraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgs*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
•
@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/ReshapeReshape<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape*
T0*
Tshape0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
: 
 
>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum_1SumDtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/Reshape_1Ptraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
”
<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/NegNeg>training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Sum_1*
T0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*
_output_shapes
:
ƒ
Btraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Reshape_1Reshape<training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Neg@training/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss/conv2d_6_loss/sub_1*1
_output_shapes
:€€€€€€€€€АА
•
!training/RMSprop/gradients/AddN_1AddNBtraining/RMSprop/gradients/loss/conv2d_6_loss/truediv_grad/ReshapeBtraining/RMSprop/gradients/loss/conv2d_6_loss/sub_1_grad/Reshape_1*
N*1
_output_shapes
:€€€€€€€€€АА*
T0*-
_class#
!loc:@loss/conv2d_6_loss/truediv
г
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ShapeShape(loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
ј
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1Const*
valueB *3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
dtype0*
_output_shapes
: 
ё
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_2Shape!training/RMSprop/gradients/AddN_1*
T0*
out_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
:
∆
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
dtype0*
_output_shapes
: 
й
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zerosFillHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_2Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros/Const*1
_output_shapes
:€€€€€€€€€АА*
T0*

index_type0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
Т
Mtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqualGreaterEqual(loss/conv2d_6_loss/clip_by_value/Minimumloss/conv2d_6_loss/Const*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
у
Vtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
ь
Gtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SelectSelectMtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqual!training/RMSprop/gradients/AddN_1Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros*1
_output_shapes
:€€€€€€€€€АА*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
ю
Itraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Select_1SelectMtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/GreaterEqualFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/zeros!training/RMSprop/gradients/AddN_1*
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
б
Dtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SumSumGtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SelectVtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value
а
Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ReshapeReshapeDtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/SumFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*1
_output_shapes
:€€€€€€€€€АА
з
Ftraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Sum_1SumItraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Select_1Xtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape_1ReshapeFtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Sum_1Htraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*3
_class)
'%loc:@loss/conv2d_6_loss/clip_by_value*
_output_shapes
: 
Ў
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ShapeShapeconv2d_6/Tanh*
T0*
out_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:
–
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 
Х
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_2ShapeHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape*
T0*
out_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
:
÷
Ttraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: 
Й
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zerosFillPtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_2Ttraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
€
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualconv2d_6/Tanhloss/conv2d_6_loss/sub*1
_output_shapes
:€€€€€€€€€АА*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
У
^training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ShapePtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
ј
Otraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SelectSelectRtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqualHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/ReshapeNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zeros*1
_output_shapes
:€€€€€€€€€АА*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
¬
Qtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select_1SelectRtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/LessEqualNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/zerosHtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value_grad/Reshape*
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*1
_output_shapes
:€€€€€€€€€АА
Б
Ltraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SumSumOtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select^training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
А
Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/ReshapeReshapeLtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/SumNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
З
Ntraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Sum_1SumQtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Select_1`training/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum
л
Rtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeNtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Sum_1Ptraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*;
_class1
/-loc:@loss/conv2d_6_loss/clip_by_value/Minimum*
_output_shapes
: 
Б
6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGradTanhGradconv2d_6/TanhPtraining/RMSprop/gradients/loss/conv2d_6_loss/clip_by_value/Minimum_grad/Reshape*
T0* 
_class
loc:@conv2d_6/Tanh*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*
T0*#
_class
loc:@conv2d_6/BiasAdd*
data_formatNHWC*
_output_shapes
:
п
;training/RMSprop/gradients/conv2d_6/convolution_grad/ShapeNShapeN%up_sampling2d_3/ResizeNearestNeighborconv2d_6/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_6/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_6/convolution_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"            *'
_class
loc:@conv2d_6/convolution
љ
Htraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_6/convolution_grad/ShapeNconv2d_6/kernel/read6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА
ƒ
Itraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_3/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_6/convolution_grad/Const6training/RMSprop/gradients/conv2d_6/Tanh_grad/TanhGrad*
	dilations
*
T0*'
_class
loc:@conv2d_6/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
п
dtraining/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"А   А   *8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
Ј
_training/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*1
_output_shapes
:€€€€€€€€€АА
м
Jtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_3/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_3/ResizeNearestNeighbor*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
щ
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_5/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
И
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/add_1*
_output_shapes
:
Й
!training/RMSprop/gradients/SwitchSwitch%batch_normalization_5/batchnorm/add_1"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
љ
"training/RMSprop/gradients/Shape_1Shape!training/RMSprop/gradients/Switch*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:
•
&training/RMSprop/gradients/zeros/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
dtype0*
_output_shapes
: 
ь
 training/RMSprop/gradients/zerosFill"training/RMSprop/gradients/Shape_1&training/RMSprop/gradients/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
«
Mtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_gradMerge training/RMSprop/gradients/zerosLtraining/RMSprop/gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
А
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_5/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1
Ы
`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape(batch_normalization_5/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
И
Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ў
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_5/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape*1
_output_shapes
:€€€€€€€€€АА*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1
М
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
_output_shapes
:
Л
Ltraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/sub
к
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ShapeShape%batch_normalization_5/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
—
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1Const*
valueB:*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1
ъ
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_5/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/add_1*
_output_shapes
:
џ
#training/RMSprop/gradients/Switch_1Switchconv2d_5/Relu"batch_normalization_5/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0* 
_class
loc:@conv2d_5/Relu
©
"training/RMSprop/gradients/Shape_2Shape%training/RMSprop/gradients/Switch_1:1*
T0*
out_type0* 
_class
loc:@conv2d_5/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_1/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_5/Relu*
dtype0*
_output_shapes
: 
и
"training/RMSprop/gradients/zeros_1Fill"training/RMSprop/gradients/Shape_2(training/RMSprop/gradients/zeros_1/Const*1
_output_shapes
:€€€€€€€€€АА*
T0*

index_type0* 
_class
loc:@conv2d_5/Relu
≈
[training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_1*
T0* 
_class
loc:@conv2d_5/Relu*
N*3
_output_shapes!
:€€€€€€€€€АА: 
ћ
#training/RMSprop/gradients/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_5/beta* 
_output_shapes
::
ґ
"training/RMSprop/gradients/Shape_3Shape%training/RMSprop/gradients/Switch_2:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:
Ь
(training/RMSprop/gradients/zeros_2/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
: 
ё
"training/RMSprop/gradients/zeros_2Fill"training/RMSprop/gradients/Shape_3(training/RMSprop/gradients/zeros_2/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:
ї
Ytraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/add_1_grad/Reshape_1"training/RMSprop/gradients/zeros_2*
T0*-
_class#
!loc:@batch_normalization_5/beta*
N*
_output_shapes

:: 
±
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/Neg(batch_normalization_5/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_2*
_output_shapes
:
Љ
Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub_grad/Neg1batch_normalization_5/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_2*
_output_shapes
:
“
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ShapeShapeconv2d_5/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
—
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
З
[training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
Ї
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape#batch_normalization_5/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
т
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1
ф
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
¶
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul_1Mulconv2d_5/ReluMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ш
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
_output_shapes
:
ь
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/sub*
_output_shapes
:
Њ
!training/RMSprop/gradients/AddN_2AddNTtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_5/cond/batchnorm/mul_1*
N*
_output_shapes
:
Й
Ltraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_2/batch_normalization_5/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/mul*
_output_shapes
:
Ж
Ntraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_2*batch_normalization_5/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_5/cond/batchnorm/mul*
_output_shapes
:
≤
!training/RMSprop/gradients/AddN_3AddNYtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_5/beta*
N*
_output_shapes
:
Э
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/Neg#batch_normalization_5/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_2*
_output_shapes
:
°
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/sub_grad/Neg%batch_normalization_5/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_2*
_output_shapes
:
ќ
#training/RMSprop/gradients/Switch_3Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma* 
_output_shapes
::
Ј
"training/RMSprop/gradients/Shape_4Shape%training/RMSprop/gradients/Switch_3:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
Э
(training/RMSprop/gradients/zeros_3/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
я
"training/RMSprop/gradients/zeros_3Fill"training/RMSprop/gradients/Shape_4(training/RMSprop/gradients/zeros_3/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:
ґ
Ytraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_grad/Mul_1"training/RMSprop/gradients/zeros_3*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
N*
_output_shapes

:: 
ё
Ktraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ShapeConst*%
valueB"            *8
_class.
,*loc:@batch_normalization_5/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_5/moments/Squeeze*&
_output_shapes
:
ѓ
!training/RMSprop/gradients/AddN_4AddNOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/mul_1*
N*
_output_shapes
:
р
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_4 batch_normalization_5/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/mul*
_output_shapes
:
ч
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_4%batch_normalization_5/batchnorm/Rsqrt*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/mul
Ђ
Otraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_5/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_5/batchnorm/Rsqrt*
_output_shapes
:
≠
!training/RMSprop/gradients/AddN_5AddNYtraining/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
N*
_output_shapes
:
Ћ
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ShapeConst*
valueB:*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add
’
Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
_output_shapes
:
ц
Itraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_5/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
„
Mtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_5/batchnorm/add
в
Mtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            *:
_class0
.,loc:@batch_normalization_5/moments/Squeeze_1
с
Otraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_5/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/Shape*&
_output_shapes
:*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_5/moments/Squeeze_1
ц
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeShape/batch_normalization_5/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/addAdd8batch_normalization_5/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
ѕ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance
≠
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@batch_normalization_5/moments/variance
ќ
Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
Ќ
Ptraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance
к
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_2Shape/batch_normalization_5/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_3Const*%
valueB"            *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
”
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
:
о
Mtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_5/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_5/moments/variance*
_output_shapes
: *

DstT0
к
Ntraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/Cast*1
_output_shapes
:€€€€€€€€€АА*
T0*9
_class/
-+loc:@batch_normalization_5/moments/variance
ж
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ShapeShapeconv2d_5/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1Const*%
valueB"            *B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference
В
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*1
_output_shapes
:€€€€€€€€€АА*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference
ж
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/subSubconv2d_5/Relu*batch_normalization_5/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_5/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ж
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ь
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ь
Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
†
Utraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference
Ч
Ytraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*&
_output_shapes
:
™
Straining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_5/moments/SquaredDifference*&
_output_shapes
:
ћ
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeShapeconv2d_5/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/addAdd4batch_normalization_5/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_5/moments/mean
∆
Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean
Ў
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean
ф
Ptraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean
≈
Ltraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_5/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ќ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_2Shapeconv2d_5/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_3Const*%
valueB"            *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_5/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_5/moments/mean*
_output_shapes
: *

DstT0
Џ
Jtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_5/moments/mean*1
_output_shapes
:€€€€€€€€€АА
б
!training/RMSprop/gradients/AddN_6AddN[training/RMSprop/gradients/batch_normalization_5/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_5/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_5/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_5/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_5/Relu*
N*1
_output_shapes
:€€€€€€€€€АА
“
6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGradReluGrad!training/RMSprop/gradients/AddN_6conv2d_5/Relu*
T0* 
_class
loc:@conv2d_5/Relu*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_5/BiasAdd
п
;training/RMSprop/gradients/conv2d_5/convolution_grad/ShapeNShapeN%up_sampling2d_2/ResizeNearestNeighborconv2d_5/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_5/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_5/convolution_grad/ConstConst*%
valueB"             *'
_class
loc:@conv2d_5/convolution*
dtype0*
_output_shapes
:
љ
Htraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_5/convolution_grad/ShapeNconv2d_5/kernel/read6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА *
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ƒ
Itraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_2/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_5/convolution_grad/Const6training/RMSprop/gradients/conv2d_5/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_5/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
п
dtraining/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"@   @   *8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
dtype0*
_output_shapes
:
µ
_training/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*/
_output_shapes
:€€€€€€€€€@@ 
и
Jtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
щ
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
џ
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
Ж
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
Ж
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
ч
Ttraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/add_1
З
#training/RMSprop/gradients/Switch_4Switch%batch_normalization_4/batchnorm/add_1"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
њ
"training/RMSprop/gradients/Shape_5Shape#training/RMSprop/gradients/Switch_4*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:
І
(training/RMSprop/gradients/zeros_4/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
dtype0*
_output_shapes
: 
ю
"training/RMSprop/gradients/zeros_4Fill"training/RMSprop/gradients/Shape_5(training/RMSprop/gradients/zeros_4/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
«
Mtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_gradMerge"training/RMSprop/gradients/zeros_4Ltraining/RMSprop/gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
N*1
_output_shapes
:€€€€€€€€€@@ : 
А
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_4/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
Ы
`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
ћ
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
Ж
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1
Ж
Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
„
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_4/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
М
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
:
ч
Ttraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
_output_shapes
: 
Л
Ltraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/sub*
_output_shapes
: 
к
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
—
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
ъ
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_4/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1
г
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/add_1*
_output_shapes
: 
„
#training/RMSprop/gradients/Switch_5Switchconv2d_4/Relu"batch_normalization_4/cond/pred_id*
T0* 
_class
loc:@conv2d_4/Relu*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
©
"training/RMSprop/gradients/Shape_6Shape%training/RMSprop/gradients/Switch_5:1*
T0*
out_type0* 
_class
loc:@conv2d_4/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_5/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_4/Relu*
dtype0*
_output_shapes
: 
ж
"training/RMSprop/gradients/zeros_5Fill"training/RMSprop/gradients/Shape_6(training/RMSprop/gradients/zeros_5/Const*
T0*

index_type0* 
_class
loc:@conv2d_4/Relu*/
_output_shapes
:€€€€€€€€€@@ 
√
[training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_5*
T0* 
_class
loc:@conv2d_4/Relu*
N*1
_output_shapes
:€€€€€€€€€@@ : 
ћ
#training/RMSprop/gradients/Switch_6Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
: : 
ґ
"training/RMSprop/gradients/Shape_7Shape%training/RMSprop/gradients/Switch_6:1*
_output_shapes
:*
T0*
out_type0*-
_class#
!loc:@batch_normalization_4/beta
Ь
(training/RMSprop/gradients/zeros_6/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
ё
"training/RMSprop/gradients/zeros_6Fill"training/RMSprop/gradients/Shape_7(training/RMSprop/gradients/zeros_6/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: 
ї
Ytraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/add_1_grad/Reshape_1"training/RMSprop/gradients/zeros_6*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*
_output_shapes

: : 
±
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg(batch_normalization_4/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes
: 
Љ
Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub_grad/Neg1batch_normalization_4/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_2*
_output_shapes
: 
“
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapeconv2d_4/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1
—
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Const*
valueB: *8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Є
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape#batch_normalization_4/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
т
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
§
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1Mulconv2d_4/ReluMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
ш
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
_output_shapes
: 
ь
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/sub*
_output_shapes
: 
Њ
!training/RMSprop/gradients/AddN_7AddNTtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_4/cond/batchnorm/mul_1*
N*
_output_shapes
: 
Й
Ltraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_7/batch_normalization_4/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes
: 
Ж
Ntraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_7*batch_normalization_4/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_4/cond/batchnorm/mul*
_output_shapes
: 
≤
!training/RMSprop/gradients/AddN_8AddNYtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
N*
_output_shapes
: 
Э
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/Neg#batch_normalization_4/batchnorm/mul*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2
°
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/sub_grad/Neg%batch_normalization_4/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_2*
_output_shapes
: 
ќ
#training/RMSprop/gradients/Switch_7Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
: : 
Ј
"training/RMSprop/gradients/Shape_8Shape%training/RMSprop/gradients/Switch_7:1*
_output_shapes
:*
T0*
out_type0*.
_class$
" loc:@batch_normalization_4/gamma
Э
(training/RMSprop/gradients/zeros_7/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
я
"training/RMSprop/gradients/zeros_7Fill"training/RMSprop/gradients/Shape_8(training/RMSprop/gradients/zeros_7/Const*
_output_shapes
: *
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma
ґ
Ytraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_grad/Mul_1"training/RMSprop/gradients/zeros_7*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
N*
_output_shapes

: : 
ё
Ktraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst*%
valueB"             *8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_4/moments/Squeeze*&
_output_shapes
: 
ѓ
!training/RMSprop/gradients/AddN_9AddNOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/mul_1*
N*
_output_shapes
: 
р
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/MulMul!training/RMSprop/gradients/AddN_9 batch_normalization_4/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes
: 
ч
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1Mul!training/RMSprop/gradients/AddN_9%batch_normalization_4/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/mul*
_output_shapes
: 
Ђ
Otraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_4/batchnorm/Rsqrt*
_output_shapes
: 
Ѓ
"training/RMSprop/gradients/AddN_10AddNYtraining/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
N*
_output_shapes
: 
Ћ
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ShapeConst*
valueB: *6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *6
_class,
*(loc:@batch_normalization_4/batchnorm/add
€
Ytraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
:
’
Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add
ц
Itraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
:
„
Mtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_4/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst*%
valueB"             *:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_4/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_4/moments/Squeeze_1*&
_output_shapes
: 
ц
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance
»
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/addAdd8batch_normalization_4/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
ќ
Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
ш
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_3Const*%
valueB"             *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance
к
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
:
о
Mtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_4/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_4/moments/variance*
_output_shapes
: *

DstT0
и
Ntraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_4/moments/variance*/
_output_shapes
:€€€€€€€€€@@ 
ж
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapeconv2d_4/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:
ф
Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Const*%
valueB"             *B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
dtype0*
_output_shapes
: 
А
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*/
_output_shapes
:€€€€€€€€€@@ *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
д
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/subSubconv2d_4/Relu*batch_normalization_4/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Д
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Ь
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
Ъ
Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference
†
Utraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*&
_output_shapes
: 
™
Straining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_4/moments/SquaredDifference*&
_output_shapes
: 
ћ
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeShapeconv2d_4/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean
®
Ftraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/addAdd4batch_normalization_4/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_4/moments/mean
∆
Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
≈
Ltraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
ё
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_4/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean
ќ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapeconv2d_4/Relu*
_output_shapes
:*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_4/moments/mean
Џ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_3Const*%
valueB"             *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_4/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*5
_class+
)'loc:@batch_normalization_4/moments/mean
Ў
Jtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_4/moments/mean*/
_output_shapes
:€€€€€€€€€@@ 
а
"training/RMSprop/gradients/AddN_11AddN[training/RMSprop/gradients/batch_normalization_4/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_4/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_4/Relu*
N*/
_output_shapes
:€€€€€€€€€@@ 
—
6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_11conv2d_4/Relu*
T0* 
_class
loc:@conv2d_4/Relu*/
_output_shapes
:€€€€€€€€€@@ 
д
<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0*#
_class
loc:@conv2d_4/BiasAdd
п
;training/RMSprop/gradients/conv2d_4/convolution_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighborconv2d_4/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_4/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_4/convolution_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"      @       *'
_class
loc:@conv2d_4/convolution
ї
Htraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_4/convolution_grad/ShapeNconv2d_4/kernel/read6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@@
ƒ
Itraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%up_sampling2d_1/ResizeNearestNeighbor:training/RMSprop/gradients/conv2d_4/convolution_grad/Const6training/RMSprop/gradients/conv2d_4/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0*'
_class
loc:@conv2d_4/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
п
dtraining/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*
_output_shapes
:*
valueB"        *8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
µ
_training/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradHtraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropInputdtraining/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*/
_output_shapes
:€€€€€€€€€  @*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
и
Jtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitch_training/RMSprop/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
щ
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_3/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
Ы
`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
Ж
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
Ж
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/add_1
З
#training/RMSprop/gradients/Switch_8Switch%batch_normalization_3/batchnorm/add_1"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
њ
"training/RMSprop/gradients/Shape_9Shape#training/RMSprop/gradients/Switch_8*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
І
(training/RMSprop/gradients/zeros_8/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
dtype0*
_output_shapes
: 
ю
"training/RMSprop/gradients/zeros_8Fill"training/RMSprop/gradients/Shape_9(training/RMSprop/gradients/zeros_8/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
«
Mtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_gradMerge"training/RMSprop/gradients/zeros_8Ltraining/RMSprop/gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
N*1
_output_shapes
:€€€€€€€€€  @: 
А
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_3/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB:@*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape(batch_normalization_3/cond/batchnorm/mul*/
_output_shapes
:€€€€€€€€€  @*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
Ж
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
Ж
Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
„
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_3/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
М
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
Л
Ltraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:@*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/sub
к
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeShape%batch_normalization_3/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1
т
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€  @
ъ
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_3/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:
г
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/add_1*
_output_shapes
:@
„
#training/RMSprop/gradients/Switch_9Switchconv2d_3/Relu"batch_normalization_3/cond/pred_id*
T0* 
_class
loc:@conv2d_3/Relu*J
_output_shapes8
6:€€€€€€€€€  @:€€€€€€€€€  @
™
#training/RMSprop/gradients/Shape_10Shape%training/RMSprop/gradients/Switch_9:1*
T0*
out_type0* 
_class
loc:@conv2d_3/Relu*
_output_shapes
:
П
(training/RMSprop/gradients/zeros_9/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2d_3/Relu
з
"training/RMSprop/gradients/zeros_9Fill#training/RMSprop/gradients/Shape_10(training/RMSprop/gradients/zeros_9/Const*
T0*

index_type0* 
_class
loc:@conv2d_3/Relu*/
_output_shapes
:€€€€€€€€€  @
√
[training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape"training/RMSprop/gradients/zeros_9*
T0* 
_class
loc:@conv2d_3/Relu*
N*1
_output_shapes
:€€€€€€€€€  @: 
Ќ
$training/RMSprop/gradients/Switch_10Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
:@:@
Є
#training/RMSprop/gradients/Shape_11Shape&training/RMSprop/gradients/Switch_10:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_10/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
б
#training/RMSprop/gradients/zeros_10Fill#training/RMSprop/gradients/Shape_11)training/RMSprop/gradients/zeros_10/Const*
_output_shapes
:@*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta
Љ
Ytraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_10*
T0*-
_class#
!loc:@batch_normalization_3/beta*
N*
_output_shapes

:@: 
±
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg(batch_normalization_3/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:@
Љ
Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub_grad/Neg1batch_normalization_3/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_2*
_output_shapes
:@
“
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeShapeconv2d_3/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
—
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1Const*
valueB:@*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Є
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
т
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
т
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
§
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1Mulconv2d_3/ReluMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€  @
ш
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
г
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1*
_output_shapes
:@
ь
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/sub*
_output_shapes
:@
њ
"training/RMSprop/gradients/AddN_12AddNTtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes
:@*
T0*=
_class3
1/loc:@batch_normalization_3/cond/batchnorm/mul_1
К
Ltraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_12/batch_normalization_3/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:@
З
Ntraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_12*batch_normalization_3/cond/batchnorm/Rsqrt*
T0*;
_class1
/-loc:@batch_normalization_3/cond/batchnorm/mul*
_output_shapes
:@
≥
"training/RMSprop/gradients/AddN_13AddNYtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_1_grad/Reshape_1*
N*
_output_shapes
:@*
T0*-
_class#
!loc:@batch_normalization_3/beta
Э
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/Neg#batch_normalization_3/batchnorm/mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes
:@
°
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/sub_grad/Neg%batch_normalization_3/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_2*
_output_shapes
:@
ѕ
$training/RMSprop/gradients/Switch_11Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
:@:@
є
#training/RMSprop/gradients/Shape_12Shape&training/RMSprop/gradients/Switch_11:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:
Ю
)training/RMSprop/gradients/zeros_11/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma
в
#training/RMSprop/gradients/zeros_11Fill#training/RMSprop/gradients/Shape_12)training/RMSprop/gradients/zeros_11/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@
Ј
Ytraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_11*
N*
_output_shapes

:@: *
T0*.
_class$
" loc:@batch_normalization_3/gamma
ё
Ktraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   *8
_class.
,*loc:@batch_normalization_3/moments/Squeeze
й
Mtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_3/moments/Squeeze*&
_output_shapes
:@
∞
"training/RMSprop/gradients/AddN_14AddNOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_2_grad/Mul_1*
N*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/mul_1
с
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_14 batch_normalization_3/gamma/read*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul
ш
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_14%batch_normalization_3/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/mul*
_output_shapes
:@
Ђ
Otraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_3/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_3/batchnorm/Rsqrt*
_output_shapes
:@
Ѓ
"training/RMSprop/gradients/AddN_15AddNYtraining/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
N*
_output_shapes
:@
Ћ
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ShapeConst*
valueB:@*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
:
∆
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add
т
Gtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgs*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
’
Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
:@
ц
Itraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_3/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
„
Mtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_3/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ShapeConst*%
valueB"         @   *:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_3/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_3/moments/Squeeze_1*&
_output_shapes
:@
ц
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeShape/batch_normalization_3/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/addAdd8batch_normalization_3/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *9
_class/
-+loc:@batch_normalization_3/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:*

Tidx0
ќ
Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance
и
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance
М
Ttraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
к
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*#
_output_shapes
:€€€€€€€€€
ў
Otraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_2Shape/batch_normalization_3/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_3Const*%
valueB"         @   *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
”
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_3/moments/variance
о
Mtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Const_1*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_3/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*9
_class/
-+loc:@batch_normalization_3/moments/variance
и
Ntraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_3/moments/variance*/
_output_shapes
:€€€€€€€€€  @
ж
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeShapeconv2d_3/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1Const*%
valueB"         @   *B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
∞
Vtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*
valueB
 *   @*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
dtype0*
_output_shapes
: 
А
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
д
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/subSubconv2d_3/Relu*batch_normalization_3/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_3/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
Д
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
Ь
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
Ъ
Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€  @
†
Utraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Shape_1*&
_output_shapes
:@*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
™
Straining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/Reshape_1*&
_output_shapes
:@*
T0*B
_class8
64loc:@batch_normalization_3/moments/SquaredDifference
ћ
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeShapeconv2d_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean
®
Ftraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/addAdd4batch_normalization_3/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
Ћ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Fill*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
N*#
_output_shapes
:€€€€€€€€€
≈
Ltraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
ё
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_3/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ќ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_2Shapeconv2d_3/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_3Const*%
valueB"         @   *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_3/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
Е
Gtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_3/moments/mean*
_output_shapes
: *

DstT0
Ў
Jtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/Cast*/
_output_shapes
:€€€€€€€€€  @*
T0*5
_class+
)'loc:@batch_normalization_3/moments/mean
а
"training/RMSprop/gradients/AddN_16AddN[training/RMSprop/gradients/batch_normalization_3/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_3/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_3/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_3/moments/mean_grad/truediv*
N*/
_output_shapes
:€€€€€€€€€  @*
T0* 
_class
loc:@conv2d_3/Relu
—
6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_16conv2d_3/Relu*
T0* 
_class
loc:@conv2d_3/Relu*/
_output_shapes
:€€€€€€€€€  @
д
<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_3/BiasAdd*
data_formatNHWC*
_output_shapes
:@
к
;training/RMSprop/gradients/conv2d_3/convolution_grad/ShapeNShapeN batch_normalization_2/cond/Mergeconv2d_3/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_3/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_3/convolution_grad/ConstConst*%
valueB"          @   *'
_class
loc:@conv2d_3/convolution*
dtype0*
_output_shapes
:
ї
Htraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_3/convolution_grad/ShapeNconv2d_3/kernel/read6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:€€€€€€€€€@@ 
њ
Itraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter batch_normalization_2/cond/Merge:training/RMSprop/gradients/conv2d_3/convolution_grad/Const6training/RMSprop/gradients/conv2d_3/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_3/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
ј
Jtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchHtraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropInput"batch_normalization_2/cond/pred_id*
T0*'
_class
loc:@conv2d_3/convolution*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
щ
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_2/cond/batchnorm/mul_1*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
Ж
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1
ч
Ttraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/add_1*
_output_shapes
: 
И
$training/RMSprop/gradients/Switch_12Switch%batch_normalization_2/batchnorm/add_1"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ 
Ѕ
#training/RMSprop/gradients/Shape_13Shape$training/RMSprop/gradients/Switch_12*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
®
)training/RMSprop/gradients/zeros_12/ConstConst*
valueB
 *    *8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
dtype0*
_output_shapes
: 
Б
#training/RMSprop/gradients/zeros_12Fill#training/RMSprop/gradients/Shape_13)training/RMSprop/gradients/zeros_12/Const*/
_output_shapes
:€€€€€€€€€@@ *
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
»
Mtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_gradMerge#training/RMSprop/gradients/zeros_12Ltraining/RMSprop/gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
N*1
_output_shapes
:€€€€€€€€€@@ : 
А
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_2/cond/batchnorm/mul_1/Switch*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:
џ
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB: *=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape(batch_normalization_2/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
Ж
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1
„
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_2/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*/
_output_shapes
:€€€€€€€€€@@ 
М
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
Ttraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
_output_shapes
: 
Л
Ltraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/sub*
_output_shapes
: 
к
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeShape%batch_normalization_2/batchnorm/mul_1*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*/
_output_shapes
:€€€€€€€€€@@ 
ъ
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_2/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1
г
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/add_1*
_output_shapes
: 
Ў
$training/RMSprop/gradients/Switch_13Switchconv2d_2/Relu"batch_normalization_2/cond/pred_id*J
_output_shapes8
6:€€€€€€€€€@@ :€€€€€€€€€@@ *
T0* 
_class
loc:@conv2d_2/Relu
Ђ
#training/RMSprop/gradients/Shape_14Shape&training/RMSprop/gradients/Switch_13:1*
T0*
out_type0* 
_class
loc:@conv2d_2/Relu*
_output_shapes
:
Р
)training/RMSprop/gradients/zeros_13/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2d_2/Relu
й
#training/RMSprop/gradients/zeros_13Fill#training/RMSprop/gradients/Shape_14)training/RMSprop/gradients/zeros_13/Const*
T0*

index_type0* 
_class
loc:@conv2d_2/Relu*/
_output_shapes
:€€€€€€€€€@@ 
ƒ
[training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape#training/RMSprop/gradients/zeros_13*
T0* 
_class
loc:@conv2d_2/Relu*
N*1
_output_shapes
:€€€€€€€€€@@ : 
Ќ
$training/RMSprop/gradients/Switch_14Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
: : 
Є
#training/RMSprop/gradients/Shape_15Shape&training/RMSprop/gradients/Switch_14:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_14/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
б
#training/RMSprop/gradients/zeros_14Fill#training/RMSprop/gradients/Shape_15)training/RMSprop/gradients/zeros_14/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
Љ
Ytraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_14*
N*
_output_shapes

: : *
T0*-
_class#
!loc:@batch_normalization_2/beta
±
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg(batch_normalization_2/cond/batchnorm/mul*
_output_shapes
: *
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2
Љ
Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub_grad/Neg1batch_normalization_2/cond/batchnorm/mul_2/Switch*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_2*
_output_shapes
: 
“
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeShapeconv2d_2/Relu*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
—
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
З
[training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Є
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape#batch_normalization_2/batchnorm/mul*/
_output_shapes
:€€€€€€€€€@@ *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
т
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:
т
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape*/
_output_shapes
:€€€€€€€€€@@ *
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
§
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1Mulconv2d_2/ReluMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape*/
_output_shapes
:€€€€€€€€€@@ *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
ш
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
г
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1
ь
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/sub*
_output_shapes
: 
њ
"training/RMSprop/gradients/AddN_17AddNTtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_2/cond/batchnorm/mul_1*
N*
_output_shapes
: 
К
Ltraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_17/batch_normalization_2/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul*
_output_shapes
: 
З
Ntraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_17*batch_normalization_2/cond/batchnorm/Rsqrt*
_output_shapes
: *
T0*;
_class1
/-loc:@batch_normalization_2/cond/batchnorm/mul
≥
"training/RMSprop/gradients/AddN_18AddNYtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
N*
_output_shapes
: 
Э
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/Neg#batch_normalization_2/batchnorm/mul*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2
°
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/sub_grad/Neg%batch_normalization_2/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_2*
_output_shapes
: 
ѕ
$training/RMSprop/gradients/Switch_15Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
: : 
є
#training/RMSprop/gradients/Shape_16Shape&training/RMSprop/gradients/Switch_15:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
Ю
)training/RMSprop/gradients/zeros_15/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
в
#training/RMSprop/gradients/zeros_15Fill#training/RMSprop/gradients/Shape_16)training/RMSprop/gradients/zeros_15/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
Ј
Ytraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_15*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
N*
_output_shapes

: : 
ё
Ktraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"             *8
_class.
,*loc:@batch_normalization_2/moments/Squeeze
й
Mtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_2/moments/Squeeze*&
_output_shapes
: 
∞
"training/RMSprop/gradients/AddN_19AddNOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/mul_1*
N*
_output_shapes
: 
с
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_19 batch_normalization_2/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes
: 
ш
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_19%batch_normalization_2/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/mul*
_output_shapes
: 
Ђ
Otraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_2/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_2/batchnorm/Rsqrt*
_output_shapes
: 
Ѓ
"training/RMSprop/gradients/AddN_20AddNYtraining/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_grad/Mul_1*
N*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_2/gamma
Ћ
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB: *6
_class,
*(loc:@batch_normalization_2/batchnorm/add
∆
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgs*
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
:*

Tidx0*
	keep_dims( 
’
Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
ц
Itraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_2/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
:
„
Mtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_2/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ShapeConst*%
valueB"             *:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_2/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/Shape*&
_output_shapes
: *
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_2/moments/Squeeze_1
ц
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeShape/batch_normalization_2/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
»
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/SizeConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
Є
Jtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/addAdd8batch_normalization_2/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
ѕ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_1Const*
valueB:*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/deltaConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
≠
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/range/delta*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:*

Tidx0
ќ
Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
и
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance
к
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
ў
Otraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
м
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/DynamicStitch*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
:
Ц
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0
ш
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_2Shape/batch_normalization_2/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_2/moments/variance
в
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_3Const*%
valueB"             *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
”
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@batch_normalization_2/moments/variance
о
Mtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_2/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1/y*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance
Ў
Qtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_2/moments/variance*
_output_shapes
: *

DstT0
и
Ntraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/Cast*
T0*9
_class/
-+loc:@batch_normalization_2/moments/variance*/
_output_shapes
:€€€€€€€€€@@ 
ж
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeShapeconv2d_2/Relu*
_output_shapes
:*
T0*
out_type0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
ф
Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1Const*%
valueB"             *B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
∞
Vtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
А
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
д
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/subSubconv2d_2/Relu*batch_normalization_2/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_2/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Д
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/sub*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
Ь
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference
Ъ
Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*/
_output_shapes
:€€€€€€€€€@@ 
†
Utraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*
_output_shapes
:
Ч
Ytraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*&
_output_shapes
: 
™
Straining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_2/moments/SquaredDifference*&
_output_shapes
: 
ћ
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeShapeconv2d_2/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/addAdd4batch_normalization_2/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Size*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
њ
Ftraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*5
_class+
)'loc:@batch_normalization_2/moments/mean
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/range/delta*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:*

Tidx0
∆
Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean
ф
Ptraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
≈
Ltraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
…
Ktraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_2/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_2/moments/mean
Ж
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean
ќ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_2Shapeconv2d_2/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_3Const*
dtype0*
_output_shapes
:*%
valueB"             *5
_class+
)'loc:@batch_normalization_2/moments/mean
…
Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ConstConst*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
Џ
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const_1Const*
valueB: *5
_class+
)'loc:@batch_normalization_2/moments/mean*
dtype0*
_output_shapes
:
ё
Itraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Const_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
«
Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*5
_class+
)'loc:@batch_normalization_2/moments/mean
 
Ltraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_2/moments/mean*
_output_shapes
: *

DstT0
Ў
Jtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/Cast*
T0*5
_class+
)'loc:@batch_normalization_2/moments/mean*/
_output_shapes
:€€€€€€€€€@@ 
а
"training/RMSprop/gradients/AddN_21AddN[training/RMSprop/gradients/batch_normalization_2/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_2/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_2/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_2/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_2/Relu*
N*/
_output_shapes
:€€€€€€€€€@@ 
—
6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_21conv2d_2/Relu*
T0* 
_class
loc:@conv2d_2/Relu*/
_output_shapes
:€€€€€€€€€@@ 
д
<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
к
;training/RMSprop/gradients/conv2d_2/convolution_grad/ShapeNShapeN batch_normalization_1/cond/Mergeconv2d_2/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_2/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_2/convolution_grad/ConstConst*%
valueB"             *'
_class
loc:@conv2d_2/convolution*
dtype0*
_output_shapes
:
љ
Htraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
њ
Itraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter batch_normalization_1/cond/Merge:training/RMSprop/gradients/conv2d_2/convolution_grad/Const6training/RMSprop/gradients/conv2d_2/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ƒ
Jtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitchHtraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput"batch_normalization_1/cond/pred_id*
T0*'
_class
loc:@conv2d_2/convolution*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
щ
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeShape*batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1
џ
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
В
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumSumJtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_grad`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
И
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
Ж
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1SumJtraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_gradbtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
ч
Ttraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/add_1*
_output_shapes
:
М
$training/RMSprop/gradients/Switch_16Switch%batch_normalization_1/batchnorm/add_1"batch_normalization_1/cond/pred_id*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
Ѕ
#training/RMSprop/gradients/Shape_17Shape$training/RMSprop/gradients/Switch_16*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
®
)training/RMSprop/gradients/zeros_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
Г
#training/RMSprop/gradients/zeros_16Fill#training/RMSprop/gradients/Shape_17)training/RMSprop/gradients/zeros_16/Const*1
_output_shapes
:€€€€€€€€€АА*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
 
Mtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_gradMerge#training/RMSprop/gradients/zeros_16Ltraining/RMSprop/gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
N*3
_output_shapes!
:€€€€€€€€€АА: 
А
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeShape1batch_normalization_1/cond/batchnorm/mul_1/Switch*
_output_shapes
:*
T0*
out_type0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
џ
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1Const*
valueB:*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
dtype0*
_output_shapes
:
Ы
`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ShapeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/MulMulRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape(batch_normalization_1/cond/batchnorm/mul*1
_output_shapes
:€€€€€€€€€АА*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
Ж
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumSumNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul`training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
И
Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/ReshapeReshapeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape*1
_output_shapes
:€€€€€€€€€АА*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
ў
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1Mul1batch_normalization_1/cond/batchnorm/mul_1/SwitchRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
М
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1SumPtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Mul_1btraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1
ч
Ttraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1ReshapePtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Sum_1Rtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
_output_shapes
:
Л
Ltraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/NegNegTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/sub*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeShape%batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
—
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
З
[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/SumSumMtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_grad[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1*1
_output_shapes
:€€€€€€€€€АА
ъ
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1SumMtraining/RMSprop/gradients/batch_normalization_1/cond/Switch_1_grad/cond_grad]training/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
г
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/add_1
№
$training/RMSprop/gradients/Switch_17Switchconv2d_1/Relu"batch_normalization_1/cond/pred_id*
T0* 
_class
loc:@conv2d_1/Relu*N
_output_shapes<
::€€€€€€€€€АА:€€€€€€€€€АА
Ђ
#training/RMSprop/gradients/Shape_18Shape&training/RMSprop/gradients/Switch_17:1*
T0*
out_type0* 
_class
loc:@conv2d_1/Relu*
_output_shapes
:
Р
)training/RMSprop/gradients/zeros_17/ConstConst*
valueB
 *    * 
_class
loc:@conv2d_1/Relu*
dtype0*
_output_shapes
: 
л
#training/RMSprop/gradients/zeros_17Fill#training/RMSprop/gradients/Shape_18)training/RMSprop/gradients/zeros_17/Const*
T0*

index_type0* 
_class
loc:@conv2d_1/Relu*1
_output_shapes
:€€€€€€€€€АА
∆
[training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMergeRtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape#training/RMSprop/gradients/zeros_17*
N*3
_output_shapes!
:€€€€€€€€€АА: *
T0* 
_class
loc:@conv2d_1/Relu
Ќ
$training/RMSprop/gradients/Switch_18Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
Є
#training/RMSprop/gradients/Shape_19Shape&training/RMSprop/gradients/Switch_18:1*
T0*
out_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
Э
)training/RMSprop/gradients/zeros_18/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
б
#training/RMSprop/gradients/zeros_18Fill#training/RMSprop/gradients/Shape_19)training/RMSprop/gradients/zeros_18/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
Љ
Ytraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub/Switch_grad/cond_gradMergeTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/add_1_grad/Reshape_1#training/RMSprop/gradients/zeros_18*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*
_output_shapes

:: 
±
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/MulMulLtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg(batch_normalization_1/cond/batchnorm/mul*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2*
_output_shapes
:
Љ
Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1MulLtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub_grad/Neg1batch_normalization_1/cond/batchnorm/mul_2/Switch*
_output_shapes
:*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_2
“
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeShapeconv2d_1/Relu*
_output_shapes
:*
T0*
out_type0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
—
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1Const*
valueB:*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
dtype0*
_output_shapes
:
З
[training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ShapeMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ї
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/MulMulMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape#batch_normalization_1/batchnorm/mul*1
_output_shapes
:€€€€€€€€€АА*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1
т
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumSumItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul[training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
¶
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1Mulconv2d_1/ReluMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*1
_output_shapes
:€€€€€€€€€АА
ш
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1SumKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Mul_1]training/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
г
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1ReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Sum_1Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
_output_shapes
:
ь
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/NegNegOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/sub
њ
"training/RMSprop/gradients/AddN_22AddNTtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1_grad/Reshape_1Ptraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_2_grad/Mul_1*
T0*=
_class3
1/loc:@batch_normalization_1/cond/batchnorm/mul_1*
N*
_output_shapes
:
К
Ltraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_22/batch_normalization_1/cond/batchnorm/mul/Switch*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul*
_output_shapes
:
З
Ntraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_22*batch_normalization_1/cond/batchnorm/Rsqrt*
_output_shapes
:*
T0*;
_class1
/-loc:@batch_normalization_1/cond/batchnorm/mul
≥
"training/RMSprop/gradients/AddN_23AddNYtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/sub/Switch_grad/cond_gradOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_1_grad/Reshape_1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
N*
_output_shapes
:
Э
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulMulGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/Neg#batch_normalization_1/batchnorm/mul*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2
°
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1MulGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/sub_grad/Neg%batch_normalization_1/moments/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_2*
_output_shapes
:
ѕ
$training/RMSprop/gradients/Switch_19Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
є
#training/RMSprop/gradients/Shape_20Shape&training/RMSprop/gradients/Switch_19:1*
T0*
out_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
Ю
)training/RMSprop/gradients/zeros_19/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
в
#training/RMSprop/gradients/zeros_19Fill#training/RMSprop/gradients/Shape_20)training/RMSprop/gradients/zeros_19/Const*
_output_shapes
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma
Ј
Ytraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul/Switch_grad/cond_gradMergeNtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_grad/Mul_1#training/RMSprop/gradients/zeros_19*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*
_output_shapes

:: 
ё
Ktraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ShapeConst*%
valueB"            *8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*
dtype0*
_output_shapes
:
й
Mtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapeReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/MulKtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/Shape*
T0*
Tshape0*8
_class.
,*loc:@batch_normalization_1/moments/Squeeze*&
_output_shapes
:
∞
"training/RMSprop/gradients/AddN_24AddNOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/Reshape_1Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_2_grad/Mul_1*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/mul_1*
N*
_output_shapes
:
с
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/MulMul"training/RMSprop/gradients/AddN_24 batch_normalization_1/gamma/read*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes
:
ш
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1Mul"training/RMSprop/gradients/AddN_24%batch_normalization_1/batchnorm/Rsqrt*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/mul*
_output_shapes
:
Ђ
Otraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_1/batchnorm/RsqrtGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul*
T0*8
_class.
,*loc:@batch_normalization_1/batchnorm/Rsqrt*
_output_shapes
:
Ѓ
"training/RMSprop/gradients/AddN_25AddNYtraining/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul/Switch_grad/cond_gradItraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_grad/Mul_1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
N*
_output_shapes
:
Ћ
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
∆
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1Const*
valueB *6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
dtype0*
_output_shapes
: 
€
Ytraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1*
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
т
Gtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/SumSumOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGradYtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
’
Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeReshapeGtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/SumItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape*
_output_shapes
:*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add
ц
Itraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Sum_1SumOtraining/RMSprop/gradients/batch_normalization_1/batchnorm/Rsqrt_grad/RsqrtGrad[training/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
:
„
Mtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Reshape_1ReshapeItraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Sum_1Ktraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/Shape_1*
T0*
Tshape0*6
_class,
*(loc:@batch_normalization_1/batchnorm/add*
_output_shapes
: 
в
Mtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ShapeConst*%
valueB"            *:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*
dtype0*
_output_shapes
:
с
Otraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeReshapeKtraining/RMSprop/gradients/batch_normalization_1/batchnorm/add_grad/ReshapeMtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*:
_class0
.,loc:@batch_normalization_1/moments/Squeeze_1*&
_output_shapes
:
ц
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeShape/batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance
»
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
Є
Jtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/addAdd8batch_normalization_1/moments/variance/reduction_indicesKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Size*
_output_shapes
:*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
ѕ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/modFloorModJtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/addKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Size*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
”
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:*9
_class/
-+loc:@batch_normalization_1/moments/variance
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/startConst*
value	B : *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
≠
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/rangeRangeRtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/startKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/SizeRtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/range/delta*

Tidx0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
ќ
Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill/valueConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
и
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/FillFillNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_1Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill/value*
T0*

index_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
М
Ttraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchDynamicStitchLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/rangeJtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/modLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Fill*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
N*#
_output_shapes
:€€€€€€€€€
Ќ
Ptraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance
к
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/MaximumMaximumTtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitchPtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
ў
Otraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordivFloorDivLtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ShapeNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
м
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ReshapeReshapeOtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_1_grad/ReshapeTtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0*9
_class/
-+loc:@batch_normalization_1/moments/variance
Ц
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/TileTileNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ReshapeOtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv*

Tmultiples0*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_2Shape/batch_normalization_1/moments/SquaredDifference*
T0*
out_type0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
:
в
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_3Const*%
valueB"            *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
—
Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ConstConst*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
к
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ProdProdNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_2Ltraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
”
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const_1Const*
valueB: *9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
:
о
Mtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Prod_1ProdNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Shape_3Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Const_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

Tidx0*
	keep_dims( 
ѕ
Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/yConst*
value	B :*9
_class/
-+loc:@batch_normalization_1/moments/variance*
dtype0*
_output_shapes
: 
Џ
Ptraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1MaximumMtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Prod_1Rtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1/y*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
Ў
Qtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv_1FloorDivKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/ProdPtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Maximum_1*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: 
С
Ktraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/CastCastQtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/floordiv_1*

SrcT0*9
_class/
-+loc:@batch_normalization_1/moments/variance*
_output_shapes
: *

DstT0
к
Ntraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truedivRealDivKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/TileKtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/Cast*1
_output_shapes
:€€€€€€€€€АА*
T0*9
_class/
-+loc:@batch_normalization_1/moments/variance
ж
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeShapeconv2d_1/Relu*
T0*
out_type0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:
ф
Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1Const*%
valueB"            *B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
dtype0*
_output_shapes
:
ѓ
etraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ShapeWtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
∞
Vtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarConstO^training/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
В
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mulMulVtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/scalarNtraining/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
ж
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/subSubconv2d_1/Relu*batch_normalization_1/moments/StopGradientO^training/RMSprop/gradients/batch_normalization_1/moments/variance_grad/truediv*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
Ж
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1MulStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mulStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/sub*1
_output_shapes
:€€€€€€€€€АА*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference
Ь
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumSumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1etraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ь
Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeReshapeStraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/SumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*1
_output_shapes
:€€€€€€€€€АА
†
Utraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1SumUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/mul_1gtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
Ytraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1ReshapeUtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Sum_1Wtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*&
_output_shapes
:
™
Straining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/NegNegYtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/Reshape_1*
T0*B
_class8
64loc:@batch_normalization_1/moments/SquaredDifference*&
_output_shapes
:
ћ
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeShapeconv2d_1/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
ј
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/SizeConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
®
Ftraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/addAdd4batch_normalization_1/moments/mean/reduction_indicesGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
њ
Ftraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/modFloorModFtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/addGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Size*
_output_shapes
:*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
Ћ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_1Const*
valueB:*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/startConst*
value	B : *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/deltaConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Щ
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/rangeRangeNtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/startGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/SizeNtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*5
_class+
)'loc:@batch_normalization_1/moments/mean
∆
Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill/valueConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Ў
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/FillFillJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_1Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill/value*
T0*

index_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
ф
Ptraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchDynamicStitchHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/rangeFtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/modHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
≈
Ltraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/MaximumMaximumPtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitchLtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*#
_output_shapes
:€€€€€€€€€
…
Ktraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordivFloorDivHtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ShapeJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
ё
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ReshapeReshapeMtraining/RMSprop/gradients/batch_normalization_1/moments/Squeeze_grad/ReshapePtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/DynamicStitch*
T0*
Tshape0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
Ж
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/TileTileJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ReshapeKtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
ќ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_2Shapeconv2d_1/Relu*
T0*
out_type0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
:
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_3Const*%
valueB"            *5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
:
…
Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean
Џ
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ProdProdJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_2Htraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
Ћ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *5
_class+
)'loc:@batch_normalization_1/moments/mean
ё
Itraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Prod_1ProdJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Shape_3Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
«
Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/yConst*
value	B :*5
_class+
)'loc:@batch_normalization_1/moments/mean*
dtype0*
_output_shapes
: 
 
Ltraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1MaximumItraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Prod_1Ntraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1/y*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
»
Mtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv_1FloorDivGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/ProdLtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Maximum_1*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: 
Е
Gtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/CastCastMtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/floordiv_1*

SrcT0*5
_class+
)'loc:@batch_normalization_1/moments/mean*
_output_shapes
: *

DstT0
Џ
Jtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/truedivRealDivGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/TileGtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/Cast*1
_output_shapes
:€€€€€€€€€АА*
T0*5
_class+
)'loc:@batch_normalization_1/moments/mean
в
"training/RMSprop/gradients/AddN_26AddN[training/RMSprop/gradients/batch_normalization_1/cond/batchnorm/mul_1/Switch_grad/cond_gradMtraining/RMSprop/gradients/batch_normalization_1/batchnorm/mul_1_grad/ReshapeWtraining/RMSprop/gradients/batch_normalization_1/moments/SquaredDifference_grad/ReshapeJtraining/RMSprop/gradients/batch_normalization_1/moments/mean_grad/truediv*
T0* 
_class
loc:@conv2d_1/Relu*
N*1
_output_shapes
:€€€€€€€€€АА
”
6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGradReluGrad"training/RMSprop/gradients/AddN_26conv2d_1/Relu*
T0* 
_class
loc:@conv2d_1/Relu*1
_output_shapes
:€€€€€€€€€АА
д
<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*#
_class
loc:@conv2d_1/BiasAdd*
data_formatNHWC*
_output_shapes
:
Ў
;training/RMSprop/gradients/conv2d_1/convolution_grad/ShapeNShapeNconv2d_1_inputconv2d_1/kernel/read*
T0*
out_type0*'
_class
loc:@conv2d_1/convolution*
N* 
_output_shapes
::
Љ
:training/RMSprop/gradients/conv2d_1/convolution_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"            *'
_class
loc:@conv2d_1/convolution
љ
Htraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput;training/RMSprop/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА
≠
Itraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input:training/RMSprop/gradients/conv2d_1/convolution_grad/Const6training/RMSprop/gradients/conv2d_1/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
{
training/RMSprop/ConstConst*%
valueB*    *
dtype0*&
_output_shapes
:
Э
training/RMSprop/Variable
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
е
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(*&
_output_shapes
:*
use_locking(
§
training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable*
T0*,
_class"
 loc:@training/RMSprop/Variable*&
_output_shapes
:
e
training/RMSprop/Const_1Const*
dtype0*
_output_shapes
:*
valueB*    
З
training/RMSprop/Variable_1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
б
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
_output_shapes
:*
T0*.
_class$
" loc:@training/RMSprop/Variable_1
e
training/RMSprop/Const_2Const*
dtype0*
_output_shapes
:*
valueB*    
З
training/RMSprop/Variable_2
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
б
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_2
Ю
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
:
e
training/RMSprop/Const_3Const*
valueB*    *
dtype0*
_output_shapes
:
З
training/RMSprop/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
б
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:
Ю
 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:
}
training/RMSprop/Const_4Const*%
valueB *    *
dtype0*&
_output_shapes
: 
Я
training/RMSprop/Variable_4
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
н
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
validate_shape(*&
_output_shapes
: *
use_locking(
™
 training/RMSprop/Variable_4/readIdentitytraining/RMSprop/Variable_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*&
_output_shapes
: 
e
training/RMSprop/Const_5Const*
dtype0*
_output_shapes
: *
valueB *    
З
training/RMSprop/Variable_5
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
б
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
validate_shape(*
_output_shapes
: 
Ю
 training/RMSprop/Variable_5/readIdentitytraining/RMSprop/Variable_5*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
e
training/RMSprop/Const_6Const*
valueB *    *
dtype0*
_output_shapes
: 
З
training/RMSprop/Variable_6
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
б
"training/RMSprop/Variable_6/AssignAssigntraining/RMSprop/Variable_6training/RMSprop/Const_6*
T0*.
_class$
" loc:@training/RMSprop/Variable_6*
validate_shape(*
_output_shapes
: *
use_locking(
Ю
 training/RMSprop/Variable_6/readIdentitytraining/RMSprop/Variable_6*
T0*.
_class$
" loc:@training/RMSprop/Variable_6*
_output_shapes
: 
e
training/RMSprop/Const_7Const*
dtype0*
_output_shapes
: *
valueB *    
З
training/RMSprop/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
б
"training/RMSprop/Variable_7/AssignAssigntraining/RMSprop/Variable_7training/RMSprop/Const_7*
T0*.
_class$
" loc:@training/RMSprop/Variable_7*
validate_shape(*
_output_shapes
: *
use_locking(
Ю
 training/RMSprop/Variable_7/readIdentitytraining/RMSprop/Variable_7*
_output_shapes
: *
T0*.
_class$
" loc:@training/RMSprop/Variable_7
}
training/RMSprop/Const_8Const*%
valueB @*    *
dtype0*&
_output_shapes
: @
Я
training/RMSprop/Variable_8
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
н
"training/RMSprop/Variable_8/AssignAssigntraining/RMSprop/Variable_8training/RMSprop/Const_8*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_8
™
 training/RMSprop/Variable_8/readIdentitytraining/RMSprop/Variable_8*&
_output_shapes
: @*
T0*.
_class$
" loc:@training/RMSprop/Variable_8
e
training/RMSprop/Const_9Const*
valueB@*    *
dtype0*
_output_shapes
:@
З
training/RMSprop/Variable_9
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
б
"training/RMSprop/Variable_9/AssignAssigntraining/RMSprop/Variable_9training/RMSprop/Const_9*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_9*
validate_shape(*
_output_shapes
:@
Ю
 training/RMSprop/Variable_9/readIdentitytraining/RMSprop/Variable_9*
T0*.
_class$
" loc:@training/RMSprop/Variable_9*
_output_shapes
:@
f
training/RMSprop/Const_10Const*
valueB@*    *
dtype0*
_output_shapes
:@
И
training/RMSprop/Variable_10
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
е
#training/RMSprop/Variable_10/AssignAssigntraining/RMSprop/Variable_10training/RMSprop/Const_10*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10*
validate_shape(*
_output_shapes
:@
°
!training/RMSprop/Variable_10/readIdentitytraining/RMSprop/Variable_10*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10*
_output_shapes
:@
f
training/RMSprop/Const_11Const*
valueB@*    *
dtype0*
_output_shapes
:@
И
training/RMSprop/Variable_11
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
е
#training/RMSprop/Variable_11/AssignAssigntraining/RMSprop/Variable_11training/RMSprop/Const_11*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11*
validate_shape(*
_output_shapes
:@
°
!training/RMSprop/Variable_11/readIdentitytraining/RMSprop/Variable_11*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11*
_output_shapes
:@
~
training/RMSprop/Const_12Const*
dtype0*&
_output_shapes
:@ *%
valueB@ *    
†
training/RMSprop/Variable_12
VariableV2*
shape:@ *
shared_name *
dtype0*&
_output_shapes
:@ *
	container 
с
#training/RMSprop/Variable_12/AssignAssigntraining/RMSprop/Variable_12training/RMSprop/Const_12*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12*
validate_shape(*&
_output_shapes
:@ 
≠
!training/RMSprop/Variable_12/readIdentitytraining/RMSprop/Variable_12*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12*&
_output_shapes
:@ 
f
training/RMSprop/Const_13Const*
valueB *    *
dtype0*
_output_shapes
: 
И
training/RMSprop/Variable_13
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
е
#training/RMSprop/Variable_13/AssignAssigntraining/RMSprop/Variable_13training/RMSprop/Const_13*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_13*
validate_shape(*
_output_shapes
: 
°
!training/RMSprop/Variable_13/readIdentitytraining/RMSprop/Variable_13*
_output_shapes
: *
T0*/
_class%
#!loc:@training/RMSprop/Variable_13
f
training/RMSprop/Const_14Const*
valueB *    *
dtype0*
_output_shapes
: 
И
training/RMSprop/Variable_14
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
е
#training/RMSprop/Variable_14/AssignAssigntraining/RMSprop/Variable_14training/RMSprop/Const_14*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
validate_shape(*
_output_shapes
: *
use_locking(
°
!training/RMSprop/Variable_14/readIdentitytraining/RMSprop/Variable_14*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
_output_shapes
: 
f
training/RMSprop/Const_15Const*
valueB *    *
dtype0*
_output_shapes
: 
И
training/RMSprop/Variable_15
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
е
#training/RMSprop/Variable_15/AssignAssigntraining/RMSprop/Variable_15training/RMSprop/Const_15*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_15
°
!training/RMSprop/Variable_15/readIdentitytraining/RMSprop/Variable_15*
T0*/
_class%
#!loc:@training/RMSprop/Variable_15*
_output_shapes
: 
~
training/RMSprop/Const_16Const*%
valueB *    *
dtype0*&
_output_shapes
: 
†
training/RMSprop/Variable_16
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
с
#training/RMSprop/Variable_16/AssignAssigntraining/RMSprop/Variable_16training/RMSprop/Const_16*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_16*
validate_shape(*&
_output_shapes
: 
≠
!training/RMSprop/Variable_16/readIdentitytraining/RMSprop/Variable_16*
T0*/
_class%
#!loc:@training/RMSprop/Variable_16*&
_output_shapes
: 
f
training/RMSprop/Const_17Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_17
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
е
#training/RMSprop/Variable_17/AssignAssigntraining/RMSprop/Variable_17training/RMSprop/Const_17*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17*
validate_shape(*
_output_shapes
:
°
!training/RMSprop/Variable_17/readIdentitytraining/RMSprop/Variable_17*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17*
_output_shapes
:
f
training/RMSprop/Const_18Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_18
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
е
#training/RMSprop/Variable_18/AssignAssigntraining/RMSprop/Variable_18training/RMSprop/Const_18*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18*
validate_shape(*
_output_shapes
:*
use_locking(
°
!training/RMSprop/Variable_18/readIdentitytraining/RMSprop/Variable_18*
_output_shapes
:*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18
f
training/RMSprop/Const_19Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_19
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
#training/RMSprop/Variable_19/AssignAssigntraining/RMSprop/Variable_19training/RMSprop/Const_19*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19*
validate_shape(*
_output_shapes
:
°
!training/RMSprop/Variable_19/readIdentitytraining/RMSprop/Variable_19*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19*
_output_shapes
:
~
training/RMSprop/Const_20Const*
dtype0*&
_output_shapes
:*%
valueB*    
†
training/RMSprop/Variable_20
VariableV2*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
с
#training/RMSprop/Variable_20/AssignAssigntraining/RMSprop/Variable_20training/RMSprop/Const_20*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*
validate_shape(*&
_output_shapes
:
≠
!training/RMSprop/Variable_20/readIdentitytraining/RMSprop/Variable_20*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*&
_output_shapes
:
f
training/RMSprop/Const_21Const*
valueB*    *
dtype0*
_output_shapes
:
И
training/RMSprop/Variable_21
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
е
#training/RMSprop/Variable_21/AssignAssigntraining/RMSprop/Variable_21training/RMSprop/Const_21*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21*
validate_shape(*
_output_shapes
:*
use_locking(
°
!training/RMSprop/Variable_21/readIdentitytraining/RMSprop/Variable_21*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21*
_output_shapes
:
b
 training/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Є
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0	*%
_class
loc:@RMSprop/iterations
~
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read*
T0*&
_output_shapes
:
[
training/RMSprop/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 
Э
training/RMSprop/SquareSquareItraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
}
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0*&
_output_shapes
:
z
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1*
T0*&
_output_shapes
:
Џ
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
use_locking(*
T0*,
_class"
 loc:@training/RMSprop/Variable*
validate_shape(*&
_output_shapes
:
™
training/RMSprop/mul_2MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
^
training/RMSprop/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_23Const*
dtype0*
_output_shapes
: *
valueB
 *  А
У
&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_23*
T0*&
_output_shapes
:
Э
training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const_22*
T0*&
_output_shapes
:
n
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0*&
_output_shapes
:
]
training/RMSprop/add_1/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 

training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y*
T0*&
_output_shapes
:
Д
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1*
T0*&
_output_shapes
:
~
training/RMSprop/sub_1Subconv2d_1/kernel/readtraining/RMSprop/truediv*
T0*&
_output_shapes
:
 
training/RMSprop/Assign_1Assignconv2d_1/kerneltraining/RMSprop/sub_1*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:
v
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes
:
]
training/RMSprop/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 
Ж
training/RMSprop/Square_1Square<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
u
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes
:
r
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
validate_shape(*
_output_shapes
:
С
training/RMSprop/mul_5MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_25Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_25*
T0*
_output_shapes
:
Х
 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_24*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
T0*
_output_shapes
:
]
training/RMSprop/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
u
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
T0*
_output_shapes
:
z
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes
:
r
training/RMSprop/sub_3Subconv2d_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes
:
Ї
training/RMSprop/Assign_3Assignconv2d_1/biastraining/RMSprop/sub_3*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:
v
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes
:
]
training/RMSprop/sub_4/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_2Square"training/RMSprop/gradients/AddN_25*
_output_shapes
:*
T0
u
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes
:
r
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_2
w
training/RMSprop/mul_8MulRMSprop/lr/read"training/RMSprop/gradients/AddN_25*
T0*
_output_shapes
:
^
training/RMSprop/Const_26Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_27Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_27*
_output_shapes
:*
T0
Х
 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_26*
_output_shapes
:*
T0
f
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes
:
]
training/RMSprop/add_5/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
_output_shapes
:*
T0
z
training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
T0*
_output_shapes
:
А
training/RMSprop/sub_5Sub batch_normalization_1/gamma/readtraining/RMSprop/truediv_2*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_5Assignbatch_normalization_1/gammatraining/RMSprop/sub_5*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
_output_shapes
:*
T0
]
training/RMSprop/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_3Square"training/RMSprop/gradients/AddN_23*
T0*
_output_shapes
:
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
x
training/RMSprop/mul_11MulRMSprop/lr/read"training/RMSprop/gradients/AddN_23*
T0*
_output_shapes
:
^
training/RMSprop/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_29Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_29*
T0*
_output_shapes
:
Х
 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_28*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:
]
training/RMSprop/add_7/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
T0*
_output_shapes
:
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
_output_shapes
:*
T0

training/RMSprop/sub_7Subbatch_normalization_1/beta/readtraining/RMSprop/truediv_3*
T0*
_output_shapes
:
‘
training/RMSprop/Assign_7Assignbatch_normalization_1/betatraining/RMSprop/sub_7*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:*
use_locking(
Г
training/RMSprop/mul_12MulRMSprop/rho/read training/RMSprop/Variable_4/read*
T0*&
_output_shapes
: 
]
training/RMSprop/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
_output_shapes
: *
T0
Я
training/RMSprop/Square_4SquareItraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
В
training/RMSprop/mul_13Multraining/RMSprop/sub_8training/RMSprop/Square_4*
T0*&
_output_shapes
: 
А
training/RMSprop/add_8Addtraining/RMSprop/mul_12training/RMSprop/mul_13*
T0*&
_output_shapes
: 
в
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
validate_shape(*&
_output_shapes
: *
use_locking(
Ђ
training/RMSprop/mul_14MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
^
training/RMSprop/Const_30Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_31Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Ч
(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_31*
T0*&
_output_shapes
: 
°
 training/RMSprop/clip_by_value_4Maximum(training/RMSprop/clip_by_value_4/Minimumtraining/RMSprop/Const_30*
T0*&
_output_shapes
: 
r
training/RMSprop/Sqrt_4Sqrt training/RMSprop/clip_by_value_4*&
_output_shapes
: *
T0
]
training/RMSprop/add_9/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Б
training/RMSprop/add_9Addtraining/RMSprop/Sqrt_4training/RMSprop/add_9/y*
T0*&
_output_shapes
: 
З
training/RMSprop/truediv_4RealDivtraining/RMSprop/mul_14training/RMSprop/add_9*
T0*&
_output_shapes
: 
А
training/RMSprop/sub_9Subconv2d_2/kernel/readtraining/RMSprop/truediv_4*
T0*&
_output_shapes
: 
 
training/RMSprop/Assign_9Assignconv2d_2/kerneltraining/RMSprop/sub_9*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel
w
training/RMSprop/mul_15MulRMSprop/rho/read training/RMSprop/Variable_5/read*
_output_shapes
: *
T0
^
training/RMSprop/sub_10/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 
Ж
training/RMSprop/Square_5Square<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
w
training/RMSprop/mul_16Multraining/RMSprop/sub_10training/RMSprop/Square_5*
_output_shapes
: *
T0
u
training/RMSprop/add_10Addtraining/RMSprop/mul_15training/RMSprop/mul_16*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
validate_shape(*
_output_shapes
: 
Т
training/RMSprop/mul_17MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
^
training/RMSprop/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_33Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_33*
_output_shapes
: *
T0
Х
 training/RMSprop/clip_by_value_5Maximum(training/RMSprop/clip_by_value_5/Minimumtraining/RMSprop/Const_32*
T0*
_output_shapes
: 
f
training/RMSprop/Sqrt_5Sqrt training/RMSprop/clip_by_value_5*
T0*
_output_shapes
: 
^
training/RMSprop/add_11/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_11Addtraining/RMSprop/Sqrt_5training/RMSprop/add_11/y*
_output_shapes
: *
T0
|
training/RMSprop/truediv_5RealDivtraining/RMSprop/mul_17training/RMSprop/add_11*
T0*
_output_shapes
: 
s
training/RMSprop/sub_11Subconv2d_2/bias/readtraining/RMSprop/truediv_5*
T0*
_output_shapes
: 
Љ
training/RMSprop/Assign_11Assignconv2d_2/biastraining/RMSprop/sub_11*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
: 
w
training/RMSprop/mul_18MulRMSprop/rho/read training/RMSprop/Variable_6/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_12Subtraining/RMSprop/sub_12/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_6Square"training/RMSprop/gradients/AddN_20*
_output_shapes
: *
T0
w
training/RMSprop/mul_19Multraining/RMSprop/sub_12training/RMSprop/Square_6*
T0*
_output_shapes
: 
u
training/RMSprop/add_12Addtraining/RMSprop/mul_18training/RMSprop/mul_19*
_output_shapes
: *
T0
Ў
training/RMSprop/Assign_12Assigntraining/RMSprop/Variable_6training/RMSprop/add_12*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_6*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_20MulRMSprop/lr/read"training/RMSprop/gradients/AddN_20*
_output_shapes
: *
T0
^
training/RMSprop/Const_34Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_35Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_6/MinimumMinimumtraining/RMSprop/add_12training/RMSprop/Const_35*
T0*
_output_shapes
: 
Х
 training/RMSprop/clip_by_value_6Maximum(training/RMSprop/clip_by_value_6/Minimumtraining/RMSprop/Const_34*
T0*
_output_shapes
: 
f
training/RMSprop/Sqrt_6Sqrt training/RMSprop/clip_by_value_6*
T0*
_output_shapes
: 
^
training/RMSprop/add_13/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_13Addtraining/RMSprop/Sqrt_6training/RMSprop/add_13/y*
T0*
_output_shapes
: 
|
training/RMSprop/truediv_6RealDivtraining/RMSprop/mul_20training/RMSprop/add_13*
T0*
_output_shapes
: 
Б
training/RMSprop/sub_13Sub batch_normalization_2/gamma/readtraining/RMSprop/truediv_6*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_13Assignbatch_normalization_2/gammatraining/RMSprop/sub_13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma
w
training/RMSprop/mul_21MulRMSprop/rho/read training/RMSprop/Variable_7/read*
_output_shapes
: *
T0
^
training/RMSprop/sub_14/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_14Subtraining/RMSprop/sub_14/xRMSprop/rho/read*
_output_shapes
: *
T0
l
training/RMSprop/Square_7Square"training/RMSprop/gradients/AddN_18*
_output_shapes
: *
T0
w
training/RMSprop/mul_22Multraining/RMSprop/sub_14training/RMSprop/Square_7*
T0*
_output_shapes
: 
u
training/RMSprop/add_14Addtraining/RMSprop/mul_21training/RMSprop/mul_22*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_14Assigntraining/RMSprop/Variable_7training/RMSprop/add_14*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_7*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_23MulRMSprop/lr/read"training/RMSprop/gradients/AddN_18*
T0*
_output_shapes
: 
^
training/RMSprop/Const_36Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_37Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_7/MinimumMinimumtraining/RMSprop/add_14training/RMSprop/Const_37*
T0*
_output_shapes
: 
Х
 training/RMSprop/clip_by_value_7Maximum(training/RMSprop/clip_by_value_7/Minimumtraining/RMSprop/Const_36*
_output_shapes
: *
T0
f
training/RMSprop/Sqrt_7Sqrt training/RMSprop/clip_by_value_7*
T0*
_output_shapes
: 
^
training/RMSprop/add_15/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_15Addtraining/RMSprop/Sqrt_7training/RMSprop/add_15/y*
T0*
_output_shapes
: 
|
training/RMSprop/truediv_7RealDivtraining/RMSprop/mul_23training/RMSprop/add_15*
T0*
_output_shapes
: 
А
training/RMSprop/sub_15Subbatch_normalization_2/beta/readtraining/RMSprop/truediv_7*
T0*
_output_shapes
: 
÷
training/RMSprop/Assign_15Assignbatch_normalization_2/betatraining/RMSprop/sub_15*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
: *
use_locking(
Г
training/RMSprop/mul_24MulRMSprop/rho/read training/RMSprop/Variable_8/read*
T0*&
_output_shapes
: @
^
training/RMSprop/sub_16/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_16Subtraining/RMSprop/sub_16/xRMSprop/rho/read*
T0*
_output_shapes
: 
Я
training/RMSprop/Square_8SquareItraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
Г
training/RMSprop/mul_25Multraining/RMSprop/sub_16training/RMSprop/Square_8*&
_output_shapes
: @*
T0
Б
training/RMSprop/add_16Addtraining/RMSprop/mul_24training/RMSprop/mul_25*
T0*&
_output_shapes
: @
д
training/RMSprop/Assign_16Assigntraining/RMSprop/Variable_8training/RMSprop/add_16*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_8
Ђ
training/RMSprop/mul_26MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_3/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
^
training/RMSprop/Const_38Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_39Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Ш
(training/RMSprop/clip_by_value_8/MinimumMinimumtraining/RMSprop/add_16training/RMSprop/Const_39*
T0*&
_output_shapes
: @
°
 training/RMSprop/clip_by_value_8Maximum(training/RMSprop/clip_by_value_8/Minimumtraining/RMSprop/Const_38*
T0*&
_output_shapes
: @
r
training/RMSprop/Sqrt_8Sqrt training/RMSprop/clip_by_value_8*
T0*&
_output_shapes
: @
^
training/RMSprop/add_17/yConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
Г
training/RMSprop/add_17Addtraining/RMSprop/Sqrt_8training/RMSprop/add_17/y*&
_output_shapes
: @*
T0
И
training/RMSprop/truediv_8RealDivtraining/RMSprop/mul_26training/RMSprop/add_17*&
_output_shapes
: @*
T0
Б
training/RMSprop/sub_17Subconv2d_3/kernel/readtraining/RMSprop/truediv_8*
T0*&
_output_shapes
: @
ћ
training/RMSprop/Assign_17Assignconv2d_3/kerneltraining/RMSprop/sub_17*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel
w
training/RMSprop/mul_27MulRMSprop/rho/read training/RMSprop/Variable_9/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_18/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_18Subtraining/RMSprop/sub_18/xRMSprop/rho/read*
T0*
_output_shapes
: 
Ж
training/RMSprop/Square_9Square<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
w
training/RMSprop/mul_28Multraining/RMSprop/sub_18training/RMSprop/Square_9*
_output_shapes
:@*
T0
u
training/RMSprop/add_18Addtraining/RMSprop/mul_27training/RMSprop/mul_28*
T0*
_output_shapes
:@
Ў
training/RMSprop/Assign_18Assigntraining/RMSprop/Variable_9training/RMSprop/add_18*
use_locking(*
T0*.
_class$
" loc:@training/RMSprop/Variable_9*
validate_shape(*
_output_shapes
:@
Т
training/RMSprop/mul_29MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
^
training/RMSprop/Const_40Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_41Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_9/MinimumMinimumtraining/RMSprop/add_18training/RMSprop/Const_41*
_output_shapes
:@*
T0
Х
 training/RMSprop/clip_by_value_9Maximum(training/RMSprop/clip_by_value_9/Minimumtraining/RMSprop/Const_40*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_9Sqrt training/RMSprop/clip_by_value_9*
_output_shapes
:@*
T0
^
training/RMSprop/add_19/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_19Addtraining/RMSprop/Sqrt_9training/RMSprop/add_19/y*
T0*
_output_shapes
:@
|
training/RMSprop/truediv_9RealDivtraining/RMSprop/mul_29training/RMSprop/add_19*
_output_shapes
:@*
T0
s
training/RMSprop/sub_19Subconv2d_3/bias/readtraining/RMSprop/truediv_9*
T0*
_output_shapes
:@
Љ
training/RMSprop/Assign_19Assignconv2d_3/biastraining/RMSprop/sub_19*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
validate_shape(*
_output_shapes
:@
x
training/RMSprop/mul_30MulRMSprop/rho/read!training/RMSprop/Variable_10/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_20/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_20Subtraining/RMSprop/sub_20/xRMSprop/rho/read*
T0*
_output_shapes
: 
m
training/RMSprop/Square_10Square"training/RMSprop/gradients/AddN_15*
T0*
_output_shapes
:@
x
training/RMSprop/mul_31Multraining/RMSprop/sub_20training/RMSprop/Square_10*
T0*
_output_shapes
:@
u
training/RMSprop/add_20Addtraining/RMSprop/mul_30training/RMSprop/mul_31*
T0*
_output_shapes
:@
Џ
training/RMSprop/Assign_20Assigntraining/RMSprop/Variable_10training/RMSprop/add_20*
T0*/
_class%
#!loc:@training/RMSprop/Variable_10*
validate_shape(*
_output_shapes
:@*
use_locking(
x
training/RMSprop/mul_32MulRMSprop/lr/read"training/RMSprop/gradients/AddN_15*
T0*
_output_shapes
:@
^
training/RMSprop/Const_42Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_43Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_10/MinimumMinimumtraining/RMSprop/add_20training/RMSprop/Const_43*
T0*
_output_shapes
:@
Ч
!training/RMSprop/clip_by_value_10Maximum)training/RMSprop/clip_by_value_10/Minimumtraining/RMSprop/Const_42*
_output_shapes
:@*
T0
h
training/RMSprop/Sqrt_10Sqrt!training/RMSprop/clip_by_value_10*
T0*
_output_shapes
:@
^
training/RMSprop/add_21/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_21Addtraining/RMSprop/Sqrt_10training/RMSprop/add_21/y*
_output_shapes
:@*
T0
}
training/RMSprop/truediv_10RealDivtraining/RMSprop/mul_32training/RMSprop/add_21*
T0*
_output_shapes
:@
В
training/RMSprop/sub_21Sub batch_normalization_3/gamma/readtraining/RMSprop/truediv_10*
_output_shapes
:@*
T0
Ў
training/RMSprop/Assign_21Assignbatch_normalization_3/gammatraining/RMSprop/sub_21*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:@
x
training/RMSprop/mul_33MulRMSprop/rho/read!training/RMSprop/Variable_11/read*
T0*
_output_shapes
:@
^
training/RMSprop/sub_22/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_22Subtraining/RMSprop/sub_22/xRMSprop/rho/read*
T0*
_output_shapes
: 
m
training/RMSprop/Square_11Square"training/RMSprop/gradients/AddN_13*
_output_shapes
:@*
T0
x
training/RMSprop/mul_34Multraining/RMSprop/sub_22training/RMSprop/Square_11*
T0*
_output_shapes
:@
u
training/RMSprop/add_22Addtraining/RMSprop/mul_33training/RMSprop/mul_34*
T0*
_output_shapes
:@
Џ
training/RMSprop/Assign_22Assigntraining/RMSprop/Variable_11training/RMSprop/add_22*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_11
x
training/RMSprop/mul_35MulRMSprop/lr/read"training/RMSprop/gradients/AddN_13*
_output_shapes
:@*
T0
^
training/RMSprop/Const_44Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_45Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_11/MinimumMinimumtraining/RMSprop/add_22training/RMSprop/Const_45*
T0*
_output_shapes
:@
Ч
!training/RMSprop/clip_by_value_11Maximum)training/RMSprop/clip_by_value_11/Minimumtraining/RMSprop/Const_44*
T0*
_output_shapes
:@
h
training/RMSprop/Sqrt_11Sqrt!training/RMSprop/clip_by_value_11*
T0*
_output_shapes
:@
^
training/RMSprop/add_23/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_23Addtraining/RMSprop/Sqrt_11training/RMSprop/add_23/y*
T0*
_output_shapes
:@
}
training/RMSprop/truediv_11RealDivtraining/RMSprop/mul_35training/RMSprop/add_23*
_output_shapes
:@*
T0
Б
training/RMSprop/sub_23Subbatch_normalization_3/beta/readtraining/RMSprop/truediv_11*
_output_shapes
:@*
T0
÷
training/RMSprop/Assign_23Assignbatch_normalization_3/betatraining/RMSprop/sub_23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta
Д
training/RMSprop/mul_36MulRMSprop/rho/read!training/RMSprop/Variable_12/read*&
_output_shapes
:@ *
T0
^
training/RMSprop/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
l
training/RMSprop/sub_24Subtraining/RMSprop/sub_24/xRMSprop/rho/read*
_output_shapes
: *
T0
†
training/RMSprop/Square_12SquareItraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
Д
training/RMSprop/mul_37Multraining/RMSprop/sub_24training/RMSprop/Square_12*
T0*&
_output_shapes
:@ 
Б
training/RMSprop/add_24Addtraining/RMSprop/mul_36training/RMSprop/mul_37*&
_output_shapes
:@ *
T0
ж
training/RMSprop/Assign_24Assigntraining/RMSprop/Variable_12training/RMSprop/add_24*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_12
Ђ
training/RMSprop/mul_38MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_4/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
^
training/RMSprop/Const_46Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_47Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_12/MinimumMinimumtraining/RMSprop/add_24training/RMSprop/Const_47*
T0*&
_output_shapes
:@ 
£
!training/RMSprop/clip_by_value_12Maximum)training/RMSprop/clip_by_value_12/Minimumtraining/RMSprop/Const_46*
T0*&
_output_shapes
:@ 
t
training/RMSprop/Sqrt_12Sqrt!training/RMSprop/clip_by_value_12*
T0*&
_output_shapes
:@ 
^
training/RMSprop/add_25/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_25Addtraining/RMSprop/Sqrt_12training/RMSprop/add_25/y*
T0*&
_output_shapes
:@ 
Й
training/RMSprop/truediv_12RealDivtraining/RMSprop/mul_38training/RMSprop/add_25*
T0*&
_output_shapes
:@ 
В
training/RMSprop/sub_25Subconv2d_4/kernel/readtraining/RMSprop/truediv_12*
T0*&
_output_shapes
:@ 
ћ
training/RMSprop/Assign_25Assignconv2d_4/kerneltraining/RMSprop/sub_25*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
x
training/RMSprop/mul_39MulRMSprop/rho/read!training/RMSprop/Variable_13/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_26/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_26Subtraining/RMSprop/sub_26/xRMSprop/rho/read*
_output_shapes
: *
T0
З
training/RMSprop/Square_13Square<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
x
training/RMSprop/mul_40Multraining/RMSprop/sub_26training/RMSprop/Square_13*
T0*
_output_shapes
: 
u
training/RMSprop/add_26Addtraining/RMSprop/mul_39training/RMSprop/mul_40*
T0*
_output_shapes
: 
Џ
training/RMSprop/Assign_26Assigntraining/RMSprop/Variable_13training/RMSprop/add_26*
T0*/
_class%
#!loc:@training/RMSprop/Variable_13*
validate_shape(*
_output_shapes
: *
use_locking(
Т
training/RMSprop/mul_41MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
^
training/RMSprop/Const_48Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_49Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_13/MinimumMinimumtraining/RMSprop/add_26training/RMSprop/Const_49*
T0*
_output_shapes
: 
Ч
!training/RMSprop/clip_by_value_13Maximum)training/RMSprop/clip_by_value_13/Minimumtraining/RMSprop/Const_48*
_output_shapes
: *
T0
h
training/RMSprop/Sqrt_13Sqrt!training/RMSprop/clip_by_value_13*
T0*
_output_shapes
: 
^
training/RMSprop/add_27/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_27Addtraining/RMSprop/Sqrt_13training/RMSprop/add_27/y*
T0*
_output_shapes
: 
}
training/RMSprop/truediv_13RealDivtraining/RMSprop/mul_41training/RMSprop/add_27*
T0*
_output_shapes
: 
t
training/RMSprop/sub_27Subconv2d_4/bias/readtraining/RMSprop/truediv_13*
T0*
_output_shapes
: 
Љ
training/RMSprop/Assign_27Assignconv2d_4/biastraining/RMSprop/sub_27*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_42MulRMSprop/rho/read!training/RMSprop/Variable_14/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_28/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_28Subtraining/RMSprop/sub_28/xRMSprop/rho/read*
T0*
_output_shapes
: 
m
training/RMSprop/Square_14Square"training/RMSprop/gradients/AddN_10*
T0*
_output_shapes
: 
x
training/RMSprop/mul_43Multraining/RMSprop/sub_28training/RMSprop/Square_14*
T0*
_output_shapes
: 
u
training/RMSprop/add_28Addtraining/RMSprop/mul_42training/RMSprop/mul_43*
T0*
_output_shapes
: 
Џ
training/RMSprop/Assign_28Assigntraining/RMSprop/Variable_14training/RMSprop/add_28*
T0*/
_class%
#!loc:@training/RMSprop/Variable_14*
validate_shape(*
_output_shapes
: *
use_locking(
x
training/RMSprop/mul_44MulRMSprop/lr/read"training/RMSprop/gradients/AddN_10*
T0*
_output_shapes
: 
^
training/RMSprop/Const_50Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_51Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_14/MinimumMinimumtraining/RMSprop/add_28training/RMSprop/Const_51*
T0*
_output_shapes
: 
Ч
!training/RMSprop/clip_by_value_14Maximum)training/RMSprop/clip_by_value_14/Minimumtraining/RMSprop/Const_50*
T0*
_output_shapes
: 
h
training/RMSprop/Sqrt_14Sqrt!training/RMSprop/clip_by_value_14*
T0*
_output_shapes
: 
^
training/RMSprop/add_29/yConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
x
training/RMSprop/add_29Addtraining/RMSprop/Sqrt_14training/RMSprop/add_29/y*
T0*
_output_shapes
: 
}
training/RMSprop/truediv_14RealDivtraining/RMSprop/mul_44training/RMSprop/add_29*
T0*
_output_shapes
: 
В
training/RMSprop/sub_29Sub batch_normalization_4/gamma/readtraining/RMSprop/truediv_14*
T0*
_output_shapes
: 
Ў
training/RMSprop/Assign_29Assignbatch_normalization_4/gammatraining/RMSprop/sub_29*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
: 
x
training/RMSprop/mul_45MulRMSprop/rho/read!training/RMSprop/Variable_15/read*
T0*
_output_shapes
: 
^
training/RMSprop/sub_30/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_30Subtraining/RMSprop/sub_30/xRMSprop/rho/read*
T0*
_output_shapes
: 
l
training/RMSprop/Square_15Square!training/RMSprop/gradients/AddN_8*
T0*
_output_shapes
: 
x
training/RMSprop/mul_46Multraining/RMSprop/sub_30training/RMSprop/Square_15*
T0*
_output_shapes
: 
u
training/RMSprop/add_30Addtraining/RMSprop/mul_45training/RMSprop/mul_46*
T0*
_output_shapes
: 
Џ
training/RMSprop/Assign_30Assigntraining/RMSprop/Variable_15training/RMSprop/add_30*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_15
w
training/RMSprop/mul_47MulRMSprop/lr/read!training/RMSprop/gradients/AddN_8*
_output_shapes
: *
T0
^
training/RMSprop/Const_52Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_53Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_15/MinimumMinimumtraining/RMSprop/add_30training/RMSprop/Const_53*
_output_shapes
: *
T0
Ч
!training/RMSprop/clip_by_value_15Maximum)training/RMSprop/clip_by_value_15/Minimumtraining/RMSprop/Const_52*
_output_shapes
: *
T0
h
training/RMSprop/Sqrt_15Sqrt!training/RMSprop/clip_by_value_15*
T0*
_output_shapes
: 
^
training/RMSprop/add_31/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_31Addtraining/RMSprop/Sqrt_15training/RMSprop/add_31/y*
T0*
_output_shapes
: 
}
training/RMSprop/truediv_15RealDivtraining/RMSprop/mul_47training/RMSprop/add_31*
T0*
_output_shapes
: 
Б
training/RMSprop/sub_31Subbatch_normalization_4/beta/readtraining/RMSprop/truediv_15*
_output_shapes
: *
T0
÷
training/RMSprop/Assign_31Assignbatch_normalization_4/betatraining/RMSprop/sub_31*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
: 
Д
training/RMSprop/mul_48MulRMSprop/rho/read!training/RMSprop/Variable_16/read*
T0*&
_output_shapes
: 
^
training/RMSprop/sub_32/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_32Subtraining/RMSprop/sub_32/xRMSprop/rho/read*
T0*
_output_shapes
: 
†
training/RMSprop/Square_16SquareItraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
Д
training/RMSprop/mul_49Multraining/RMSprop/sub_32training/RMSprop/Square_16*
T0*&
_output_shapes
: 
Б
training/RMSprop/add_32Addtraining/RMSprop/mul_48training/RMSprop/mul_49*
T0*&
_output_shapes
: 
ж
training/RMSprop/Assign_32Assigntraining/RMSprop/Variable_16training/RMSprop/add_32*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_16
Ђ
training/RMSprop/mul_50MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_5/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
^
training/RMSprop/Const_54Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_55Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_16/MinimumMinimumtraining/RMSprop/add_32training/RMSprop/Const_55*
T0*&
_output_shapes
: 
£
!training/RMSprop/clip_by_value_16Maximum)training/RMSprop/clip_by_value_16/Minimumtraining/RMSprop/Const_54*
T0*&
_output_shapes
: 
t
training/RMSprop/Sqrt_16Sqrt!training/RMSprop/clip_by_value_16*
T0*&
_output_shapes
: 
^
training/RMSprop/add_33/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_33Addtraining/RMSprop/Sqrt_16training/RMSprop/add_33/y*
T0*&
_output_shapes
: 
Й
training/RMSprop/truediv_16RealDivtraining/RMSprop/mul_50training/RMSprop/add_33*
T0*&
_output_shapes
: 
В
training/RMSprop/sub_33Subconv2d_5/kernel/readtraining/RMSprop/truediv_16*
T0*&
_output_shapes
: 
ћ
training/RMSprop/Assign_33Assignconv2d_5/kerneltraining/RMSprop/sub_33*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
x
training/RMSprop/mul_51MulRMSprop/rho/read!training/RMSprop/Variable_17/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_34/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_34Subtraining/RMSprop/sub_34/xRMSprop/rho/read*
T0*
_output_shapes
: 
З
training/RMSprop/Square_17Square<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
x
training/RMSprop/mul_52Multraining/RMSprop/sub_34training/RMSprop/Square_17*
T0*
_output_shapes
:
u
training/RMSprop/add_34Addtraining/RMSprop/mul_51training/RMSprop/mul_52*
T0*
_output_shapes
:
Џ
training/RMSprop/Assign_34Assigntraining/RMSprop/Variable_17training/RMSprop/add_34*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_17*
validate_shape(*
_output_shapes
:
Т
training/RMSprop/mul_53MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
^
training/RMSprop/Const_56Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_57Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_17/MinimumMinimumtraining/RMSprop/add_34training/RMSprop/Const_57*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_17Maximum)training/RMSprop/clip_by_value_17/Minimumtraining/RMSprop/Const_56*
_output_shapes
:*
T0
h
training/RMSprop/Sqrt_17Sqrt!training/RMSprop/clip_by_value_17*
T0*
_output_shapes
:
^
training/RMSprop/add_35/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_35Addtraining/RMSprop/Sqrt_17training/RMSprop/add_35/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_17RealDivtraining/RMSprop/mul_53training/RMSprop/add_35*
T0*
_output_shapes
:
t
training/RMSprop/sub_35Subconv2d_5/bias/readtraining/RMSprop/truediv_17*
T0*
_output_shapes
:
Љ
training/RMSprop/Assign_35Assignconv2d_5/biastraining/RMSprop/sub_35*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:
x
training/RMSprop/mul_54MulRMSprop/rho/read!training/RMSprop/Variable_18/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_36/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_36Subtraining/RMSprop/sub_36/xRMSprop/rho/read*
_output_shapes
: *
T0
l
training/RMSprop/Square_18Square!training/RMSprop/gradients/AddN_5*
T0*
_output_shapes
:
x
training/RMSprop/mul_55Multraining/RMSprop/sub_36training/RMSprop/Square_18*
_output_shapes
:*
T0
u
training/RMSprop/add_36Addtraining/RMSprop/mul_54training/RMSprop/mul_55*
T0*
_output_shapes
:
Џ
training/RMSprop/Assign_36Assigntraining/RMSprop/Variable_18training/RMSprop/add_36*
T0*/
_class%
#!loc:@training/RMSprop/Variable_18*
validate_shape(*
_output_shapes
:*
use_locking(
w
training/RMSprop/mul_56MulRMSprop/lr/read!training/RMSprop/gradients/AddN_5*
T0*
_output_shapes
:
^
training/RMSprop/Const_58Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_59Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_18/MinimumMinimumtraining/RMSprop/add_36training/RMSprop/Const_59*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_18Maximum)training/RMSprop/clip_by_value_18/Minimumtraining/RMSprop/Const_58*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_18Sqrt!training/RMSprop/clip_by_value_18*
T0*
_output_shapes
:
^
training/RMSprop/add_37/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_37Addtraining/RMSprop/Sqrt_18training/RMSprop/add_37/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_18RealDivtraining/RMSprop/mul_56training/RMSprop/add_37*
T0*
_output_shapes
:
В
training/RMSprop/sub_37Sub batch_normalization_5/gamma/readtraining/RMSprop/truediv_18*
T0*
_output_shapes
:
Ў
training/RMSprop/Assign_37Assignbatch_normalization_5/gammatraining/RMSprop/sub_37*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
x
training/RMSprop/mul_57MulRMSprop/rho/read!training/RMSprop/Variable_19/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_38/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_38Subtraining/RMSprop/sub_38/xRMSprop/rho/read*
_output_shapes
: *
T0
l
training/RMSprop/Square_19Square!training/RMSprop/gradients/AddN_3*
T0*
_output_shapes
:
x
training/RMSprop/mul_58Multraining/RMSprop/sub_38training/RMSprop/Square_19*
T0*
_output_shapes
:
u
training/RMSprop/add_38Addtraining/RMSprop/mul_57training/RMSprop/mul_58*
T0*
_output_shapes
:
Џ
training/RMSprop/Assign_38Assigntraining/RMSprop/Variable_19training/RMSprop/add_38*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_19
w
training/RMSprop/mul_59MulRMSprop/lr/read!training/RMSprop/gradients/AddN_3*
T0*
_output_shapes
:
^
training/RMSprop/Const_60Const*
dtype0*
_output_shapes
: *
valueB
 *    
^
training/RMSprop/Const_61Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Н
)training/RMSprop/clip_by_value_19/MinimumMinimumtraining/RMSprop/add_38training/RMSprop/Const_61*
T0*
_output_shapes
:
Ч
!training/RMSprop/clip_by_value_19Maximum)training/RMSprop/clip_by_value_19/Minimumtraining/RMSprop/Const_60*
_output_shapes
:*
T0
h
training/RMSprop/Sqrt_19Sqrt!training/RMSprop/clip_by_value_19*
_output_shapes
:*
T0
^
training/RMSprop/add_39/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_39Addtraining/RMSprop/Sqrt_19training/RMSprop/add_39/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_19RealDivtraining/RMSprop/mul_59training/RMSprop/add_39*
_output_shapes
:*
T0
Б
training/RMSprop/sub_39Subbatch_normalization_5/beta/readtraining/RMSprop/truediv_19*
T0*
_output_shapes
:
÷
training/RMSprop/Assign_39Assignbatch_normalization_5/betatraining/RMSprop/sub_39*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes
:
Д
training/RMSprop/mul_60MulRMSprop/rho/read!training/RMSprop/Variable_20/read*&
_output_shapes
:*
T0
^
training/RMSprop/sub_40/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_40Subtraining/RMSprop/sub_40/xRMSprop/rho/read*
_output_shapes
: *
T0
†
training/RMSprop/Square_20SquareItraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
Д
training/RMSprop/mul_61Multraining/RMSprop/sub_40training/RMSprop/Square_20*
T0*&
_output_shapes
:
Б
training/RMSprop/add_40Addtraining/RMSprop/mul_60training/RMSprop/mul_61*
T0*&
_output_shapes
:
ж
training/RMSprop/Assign_40Assigntraining/RMSprop/Variable_20training/RMSprop/add_40*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_20*
validate_shape(*&
_output_shapes
:
Ђ
training/RMSprop/mul_62MulRMSprop/lr/readItraining/RMSprop/gradients/conv2d_6/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
^
training/RMSprop/Const_62Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_63Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Щ
)training/RMSprop/clip_by_value_20/MinimumMinimumtraining/RMSprop/add_40training/RMSprop/Const_63*&
_output_shapes
:*
T0
£
!training/RMSprop/clip_by_value_20Maximum)training/RMSprop/clip_by_value_20/Minimumtraining/RMSprop/Const_62*&
_output_shapes
:*
T0
t
training/RMSprop/Sqrt_20Sqrt!training/RMSprop/clip_by_value_20*
T0*&
_output_shapes
:
^
training/RMSprop/add_41/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Д
training/RMSprop/add_41Addtraining/RMSprop/Sqrt_20training/RMSprop/add_41/y*
T0*&
_output_shapes
:
Й
training/RMSprop/truediv_20RealDivtraining/RMSprop/mul_62training/RMSprop/add_41*&
_output_shapes
:*
T0
В
training/RMSprop/sub_41Subconv2d_6/kernel/readtraining/RMSprop/truediv_20*
T0*&
_output_shapes
:
ћ
training/RMSprop/Assign_41Assignconv2d_6/kerneltraining/RMSprop/sub_41*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:
x
training/RMSprop/mul_63MulRMSprop/rho/read!training/RMSprop/Variable_21/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_42/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_42Subtraining/RMSprop/sub_42/xRMSprop/rho/read*
T0*
_output_shapes
: 
З
training/RMSprop/Square_21Square<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
x
training/RMSprop/mul_64Multraining/RMSprop/sub_42training/RMSprop/Square_21*
T0*
_output_shapes
:
u
training/RMSprop/add_42Addtraining/RMSprop/mul_63training/RMSprop/mul_64*
T0*
_output_shapes
:
Џ
training/RMSprop/Assign_42Assigntraining/RMSprop/Variable_21training/RMSprop/add_42*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*/
_class%
#!loc:@training/RMSprop/Variable_21
Т
training/RMSprop/mul_65MulRMSprop/lr/read<training/RMSprop/gradients/conv2d_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_64Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_65Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
)training/RMSprop/clip_by_value_21/MinimumMinimumtraining/RMSprop/add_42training/RMSprop/Const_65*
_output_shapes
:*
T0
Ч
!training/RMSprop/clip_by_value_21Maximum)training/RMSprop/clip_by_value_21/Minimumtraining/RMSprop/Const_64*
T0*
_output_shapes
:
h
training/RMSprop/Sqrt_21Sqrt!training/RMSprop/clip_by_value_21*
T0*
_output_shapes
:
^
training/RMSprop/add_43/yConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
x
training/RMSprop/add_43Addtraining/RMSprop/Sqrt_21training/RMSprop/add_43/y*
T0*
_output_shapes
:
}
training/RMSprop/truediv_21RealDivtraining/RMSprop/mul_65training/RMSprop/add_43*
T0*
_output_shapes
:
t
training/RMSprop/sub_43Subconv2d_6/bias/readtraining/RMSprop/truediv_21*
T0*
_output_shapes
:
Љ
training/RMSprop/Assign_43Assignconv2d_6/biastraining/RMSprop/sub_43*
use_locking(*
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
Ќ
training/group_depsNoOp	^loss/mul&^batch_normalization_1/AssignMovingAvg(^batch_normalization_1/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11^training/RMSprop/Assign_12^training/RMSprop/Assign_13^training/RMSprop/Assign_14^training/RMSprop/Assign_15^training/RMSprop/Assign_16^training/RMSprop/Assign_17^training/RMSprop/Assign_18^training/RMSprop/Assign_19^training/RMSprop/Assign_20^training/RMSprop/Assign_21^training/RMSprop/Assign_22^training/RMSprop/Assign_23^training/RMSprop/Assign_24^training/RMSprop/Assign_25^training/RMSprop/Assign_26^training/RMSprop/Assign_27^training/RMSprop/Assign_28^training/RMSprop/Assign_29^training/RMSprop/Assign_30^training/RMSprop/Assign_31^training/RMSprop/Assign_32^training/RMSprop/Assign_33^training/RMSprop/Assign_34^training/RMSprop/Assign_35^training/RMSprop/Assign_36^training/RMSprop/Assign_37^training/RMSprop/Assign_38^training/RMSprop/Assign_39^training/RMSprop/Assign_40^training/RMSprop/Assign_41^training/RMSprop/Assign_42^training/RMSprop/Assign_43


group_depsNoOp	^loss/mul
И
IsVariableInitializedIsVariableInitializedconv2d_1/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
Ж
IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Ґ
IsVariableInitialized_2IsVariableInitializedbatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
†
IsVariableInitialized_3IsVariableInitializedbatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
Ѓ
IsVariableInitialized_4IsVariableInitialized!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
ґ
IsVariableInitialized_5IsVariableInitialized%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance
К
IsVariableInitialized_6IsVariableInitializedconv2d_2/kernel*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel
Ж
IsVariableInitialized_7IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 
Ґ
IsVariableInitialized_8IsVariableInitializedbatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
†
IsVariableInitialized_9IsVariableInitializedbatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
ѓ
IsVariableInitialized_10IsVariableInitialized!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_11IsVariableInitialized%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_12IsVariableInitializedconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_13IsVariableInitializedconv2d_3/bias*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_3/bias
£
IsVariableInitialized_14IsVariableInitializedbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
°
IsVariableInitialized_15IsVariableInitializedbatch_normalization_3/beta*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_3/beta
ѓ
IsVariableInitialized_16IsVariableInitialized!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_17IsVariableInitialized%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_18IsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_19IsVariableInitializedconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_20IsVariableInitializedbatch_normalization_4/gamma*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_4/gamma
°
IsVariableInitialized_21IsVariableInitializedbatch_normalization_4/beta*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_4/beta
ѓ
IsVariableInitialized_22IsVariableInitialized!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_23IsVariableInitialized%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_24IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_25IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_26IsVariableInitializedbatch_normalization_5/gamma*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
°
IsVariableInitialized_27IsVariableInitializedbatch_normalization_5/beta*
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_5/beta
ѓ
IsVariableInitialized_28IsVariableInitialized!batch_normalization_5/moving_mean*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
Ј
IsVariableInitialized_29IsVariableInitialized%batch_normalization_5/moving_variance*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_30IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_31IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 
Б
IsVariableInitialized_32IsVariableInitialized
RMSprop/lr*
_class
loc:@RMSprop/lr*
dtype0*
_output_shapes
: 
Г
IsVariableInitialized_33IsVariableInitializedRMSprop/rho*
_class
loc:@RMSprop/rho*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_34IsVariableInitializedRMSprop/decay*
dtype0*
_output_shapes
: * 
_class
loc:@RMSprop/decay
С
IsVariableInitialized_35IsVariableInitializedRMSprop/iterations*%
_class
loc:@RMSprop/iterations*
dtype0	*
_output_shapes
: 
Я
IsVariableInitialized_36IsVariableInitializedtraining/RMSprop/Variable*,
_class"
 loc:@training/RMSprop/Variable*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_37IsVariableInitializedtraining/RMSprop/Variable_1*.
_class$
" loc:@training/RMSprop/Variable_1*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_38IsVariableInitializedtraining/RMSprop/Variable_2*.
_class$
" loc:@training/RMSprop/Variable_2*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_39IsVariableInitializedtraining/RMSprop/Variable_3*.
_class$
" loc:@training/RMSprop/Variable_3*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_40IsVariableInitializedtraining/RMSprop/Variable_4*.
_class$
" loc:@training/RMSprop/Variable_4*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_41IsVariableInitializedtraining/RMSprop/Variable_5*.
_class$
" loc:@training/RMSprop/Variable_5*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_42IsVariableInitializedtraining/RMSprop/Variable_6*
dtype0*
_output_shapes
: *.
_class$
" loc:@training/RMSprop/Variable_6
£
IsVariableInitialized_43IsVariableInitializedtraining/RMSprop/Variable_7*.
_class$
" loc:@training/RMSprop/Variable_7*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_44IsVariableInitializedtraining/RMSprop/Variable_8*.
_class$
" loc:@training/RMSprop/Variable_8*
dtype0*
_output_shapes
: 
£
IsVariableInitialized_45IsVariableInitializedtraining/RMSprop/Variable_9*
dtype0*
_output_shapes
: *.
_class$
" loc:@training/RMSprop/Variable_9
•
IsVariableInitialized_46IsVariableInitializedtraining/RMSprop/Variable_10*/
_class%
#!loc:@training/RMSprop/Variable_10*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_47IsVariableInitializedtraining/RMSprop/Variable_11*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_11
•
IsVariableInitialized_48IsVariableInitializedtraining/RMSprop/Variable_12*/
_class%
#!loc:@training/RMSprop/Variable_12*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_49IsVariableInitializedtraining/RMSprop/Variable_13*/
_class%
#!loc:@training/RMSprop/Variable_13*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_50IsVariableInitializedtraining/RMSprop/Variable_14*/
_class%
#!loc:@training/RMSprop/Variable_14*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_51IsVariableInitializedtraining/RMSprop/Variable_15*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_15
•
IsVariableInitialized_52IsVariableInitializedtraining/RMSprop/Variable_16*/
_class%
#!loc:@training/RMSprop/Variable_16*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_53IsVariableInitializedtraining/RMSprop/Variable_17*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_17
•
IsVariableInitialized_54IsVariableInitializedtraining/RMSprop/Variable_18*/
_class%
#!loc:@training/RMSprop/Variable_18*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_55IsVariableInitializedtraining/RMSprop/Variable_19*/
_class%
#!loc:@training/RMSprop/Variable_19*
dtype0*
_output_shapes
: 
•
IsVariableInitialized_56IsVariableInitializedtraining/RMSprop/Variable_20*
dtype0*
_output_shapes
: */
_class%
#!loc:@training/RMSprop/Variable_20
•
IsVariableInitialized_57IsVariableInitializedtraining/RMSprop/Variable_21*/
_class%
#!loc:@training/RMSprop/Variable_21*
dtype0*
_output_shapes
: 
п
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign#^batch_normalization_1/gamma/Assign"^batch_normalization_1/beta/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign#^batch_normalization_2/gamma/Assign"^batch_normalization_2/beta/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^conv2d_3/kernel/Assign^conv2d_3/bias/Assign#^batch_normalization_3/gamma/Assign"^batch_normalization_3/beta/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign^conv2d_4/kernel/Assign^conv2d_4/bias/Assign#^batch_normalization_4/gamma/Assign"^batch_normalization_4/beta/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign^conv2d_5/kernel/Assign^conv2d_5/bias/Assign#^batch_normalization_5/gamma/Assign"^batch_normalization_5/beta/Assign)^batch_normalization_5/moving_mean/Assign-^batch_normalization_5/moving_variance/Assign^conv2d_6/kernel/Assign^conv2d_6/bias/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^RMSprop/decay/Assign^RMSprop/iterations/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign#^training/RMSprop/Variable_4/Assign#^training/RMSprop/Variable_5/Assign#^training/RMSprop/Variable_6/Assign#^training/RMSprop/Variable_7/Assign#^training/RMSprop/Variable_8/Assign#^training/RMSprop/Variable_9/Assign$^training/RMSprop/Variable_10/Assign$^training/RMSprop/Variable_11/Assign$^training/RMSprop/Variable_12/Assign$^training/RMSprop/Variable_13/Assign$^training/RMSprop/Variable_14/Assign$^training/RMSprop/Variable_15/Assign$^training/RMSprop/Variable_16/Assign$^training/RMSprop/Variable_17/Assign$^training/RMSprop/Variable_18/Assign$^training/RMSprop/Variable_19/Assign$^training/RMSprop/Variable_20/Assign$^training/RMSprop/Variable_21/Assign""Ч:
trainable_variables€9ь9
^
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:0
O
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:0
Ж
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02batch_normalization_1/Const:0
Е
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02batch_normalization_1/Const_1:0
Ъ
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:02batch_normalization_1/Const_2:0
¶
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:02batch_normalization_1/Const_3:0
^
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:0
O
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:0
Ж
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02batch_normalization_2/Const:0
Е
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02batch_normalization_2/Const_1:0
Ъ
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:02batch_normalization_2/Const_2:0
¶
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:02batch_normalization_2/Const_3:0
^
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02conv2d_3/random_uniform:0
O
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02conv2d_3/Const:0
Ж
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02batch_normalization_3/Const:0
Е
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02batch_normalization_3/Const_1:0
Ъ
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:02batch_normalization_3/Const_2:0
¶
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:02batch_normalization_3/Const_3:0
^
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/random_uniform:0
O
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:0
Ж
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:02batch_normalization_4/Const:0
Е
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:02batch_normalization_4/Const_1:0
Ъ
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign(batch_normalization_4/moving_mean/read:02batch_normalization_4/Const_2:0
¶
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign,batch_normalization_4/moving_variance/read:02batch_normalization_4/Const_3:0
^
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/random_uniform:0
O
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:0
Ж
batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign"batch_normalization_5/gamma/read:02batch_normalization_5/Const:0
Е
batch_normalization_5/beta:0!batch_normalization_5/beta/Assign!batch_normalization_5/beta/read:02batch_normalization_5/Const_1:0
Ъ
#batch_normalization_5/moving_mean:0(batch_normalization_5/moving_mean/Assign(batch_normalization_5/moving_mean/read:02batch_normalization_5/Const_2:0
¶
'batch_normalization_5/moving_variance:0,batch_normalization_5/moving_variance/Assign,batch_normalization_5/moving_variance/read:02batch_normalization_5/Const_3:0
^
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/random_uniform:0
O
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:0
P
RMSprop/lr:0RMSprop/lr/AssignRMSprop/lr/read:02RMSprop/lr/initial_value:0
T
RMSprop/rho:0RMSprop/rho/AssignRMSprop/rho/read:02RMSprop/rho/initial_value:0
\
RMSprop/decay:0RMSprop/decay/AssignRMSprop/decay/read:02RMSprop/decay/initial_value:0
p
RMSprop/iterations:0RMSprop/iterations/AssignRMSprop/iterations/read:02"RMSprop/iterations/initial_value:0
{
training/RMSprop/Variable:0 training/RMSprop/Variable/Assign training/RMSprop/Variable/read:02training/RMSprop/Const:0
Г
training/RMSprop/Variable_1:0"training/RMSprop/Variable_1/Assign"training/RMSprop/Variable_1/read:02training/RMSprop/Const_1:0
Г
training/RMSprop/Variable_2:0"training/RMSprop/Variable_2/Assign"training/RMSprop/Variable_2/read:02training/RMSprop/Const_2:0
Г
training/RMSprop/Variable_3:0"training/RMSprop/Variable_3/Assign"training/RMSprop/Variable_3/read:02training/RMSprop/Const_3:0
Г
training/RMSprop/Variable_4:0"training/RMSprop/Variable_4/Assign"training/RMSprop/Variable_4/read:02training/RMSprop/Const_4:0
Г
training/RMSprop/Variable_5:0"training/RMSprop/Variable_5/Assign"training/RMSprop/Variable_5/read:02training/RMSprop/Const_5:0
Г
training/RMSprop/Variable_6:0"training/RMSprop/Variable_6/Assign"training/RMSprop/Variable_6/read:02training/RMSprop/Const_6:0
Г
training/RMSprop/Variable_7:0"training/RMSprop/Variable_7/Assign"training/RMSprop/Variable_7/read:02training/RMSprop/Const_7:0
Г
training/RMSprop/Variable_8:0"training/RMSprop/Variable_8/Assign"training/RMSprop/Variable_8/read:02training/RMSprop/Const_8:0
Г
training/RMSprop/Variable_9:0"training/RMSprop/Variable_9/Assign"training/RMSprop/Variable_9/read:02training/RMSprop/Const_9:0
З
training/RMSprop/Variable_10:0#training/RMSprop/Variable_10/Assign#training/RMSprop/Variable_10/read:02training/RMSprop/Const_10:0
З
training/RMSprop/Variable_11:0#training/RMSprop/Variable_11/Assign#training/RMSprop/Variable_11/read:02training/RMSprop/Const_11:0
З
training/RMSprop/Variable_12:0#training/RMSprop/Variable_12/Assign#training/RMSprop/Variable_12/read:02training/RMSprop/Const_12:0
З
training/RMSprop/Variable_13:0#training/RMSprop/Variable_13/Assign#training/RMSprop/Variable_13/read:02training/RMSprop/Const_13:0
З
training/RMSprop/Variable_14:0#training/RMSprop/Variable_14/Assign#training/RMSprop/Variable_14/read:02training/RMSprop/Const_14:0
З
training/RMSprop/Variable_15:0#training/RMSprop/Variable_15/Assign#training/RMSprop/Variable_15/read:02training/RMSprop/Const_15:0
З
training/RMSprop/Variable_16:0#training/RMSprop/Variable_16/Assign#training/RMSprop/Variable_16/read:02training/RMSprop/Const_16:0
З
training/RMSprop/Variable_17:0#training/RMSprop/Variable_17/Assign#training/RMSprop/Variable_17/read:02training/RMSprop/Const_17:0
З
training/RMSprop/Variable_18:0#training/RMSprop/Variable_18/Assign#training/RMSprop/Variable_18/read:02training/RMSprop/Const_18:0
З
training/RMSprop/Variable_19:0#training/RMSprop/Variable_19/Assign#training/RMSprop/Variable_19/read:02training/RMSprop/Const_19:0
З
training/RMSprop/Variable_20:0#training/RMSprop/Variable_20/Assign#training/RMSprop/Variable_20/read:02training/RMSprop/Const_20:0
З
training/RMSprop/Variable_21:0#training/RMSprop/Variable_21/Assign#training/RMSprop/Variable_21/read:02training/RMSprop/Const_21:0"њT
cond_contextЃTЂT
ђ
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *і
'batch_normalization_1/batchnorm/add_1:0
%batch_normalization_1/cond/Switch_1:0
%batch_normalization_1/cond/Switch_1:1
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0N
%batch_normalization_1/cond/switch_t:0%batch_normalization_1/cond/switch_t:0P
'batch_normalization_1/batchnorm/add_1:0%batch_normalization_1/cond/Switch_1:1
љ
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*≈
!batch_normalization_1/beta/read:0
,batch_normalization_1/cond/batchnorm/Rsqrt:0
1batch_normalization_1/cond/batchnorm/add/Switch:0
,batch_normalization_1/cond/batchnorm/add/y:0
*batch_normalization_1/cond/batchnorm/add:0
,batch_normalization_1/cond/batchnorm/add_1:0
1batch_normalization_1/cond/batchnorm/mul/Switch:0
*batch_normalization_1/cond/batchnorm/mul:0
3batch_normalization_1/cond/batchnorm/mul_1/Switch:0
,batch_normalization_1/cond/batchnorm/mul_1:0
3batch_normalization_1/cond/batchnorm/mul_2/Switch:0
,batch_normalization_1/cond/batchnorm/mul_2:0
1batch_normalization_1/cond/batchnorm/sub/Switch:0
*batch_normalization_1/cond/batchnorm/sub:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
"batch_normalization_1/gamma/read:0
(batch_normalization_1/moving_mean/read:0
,batch_normalization_1/moving_variance/read:0
conv2d_1/Relu:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0F
conv2d_1/Relu:03batch_normalization_1/cond/batchnorm/mul_1/Switch:0N
%batch_normalization_1/cond/switch_f:0%batch_normalization_1/cond/switch_f:0_
(batch_normalization_1/moving_mean/read:03batch_normalization_1/cond/batchnorm/mul_2/Switch:0a
,batch_normalization_1/moving_variance/read:01batch_normalization_1/cond/batchnorm/add/Switch:0W
"batch_normalization_1/gamma/read:01batch_normalization_1/cond/batchnorm/mul/Switch:0V
!batch_normalization_1/beta/read:01batch_normalization_1/cond/batchnorm/sub/Switch:0
ђ
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *і
'batch_normalization_2/batchnorm/add_1:0
%batch_normalization_2/cond/Switch_1:0
%batch_normalization_2/cond/Switch_1:1
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0P
'batch_normalization_2/batchnorm/add_1:0%batch_normalization_2/cond/Switch_1:1L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0N
%batch_normalization_2/cond/switch_t:0%batch_normalization_2/cond/switch_t:0
љ
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*≈
!batch_normalization_2/beta/read:0
,batch_normalization_2/cond/batchnorm/Rsqrt:0
1batch_normalization_2/cond/batchnorm/add/Switch:0
,batch_normalization_2/cond/batchnorm/add/y:0
*batch_normalization_2/cond/batchnorm/add:0
,batch_normalization_2/cond/batchnorm/add_1:0
1batch_normalization_2/cond/batchnorm/mul/Switch:0
*batch_normalization_2/cond/batchnorm/mul:0
3batch_normalization_2/cond/batchnorm/mul_1/Switch:0
,batch_normalization_2/cond/batchnorm/mul_1:0
3batch_normalization_2/cond/batchnorm/mul_2/Switch:0
,batch_normalization_2/cond/batchnorm/mul_2:0
1batch_normalization_2/cond/batchnorm/sub/Switch:0
*batch_normalization_2/cond/batchnorm/sub:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
"batch_normalization_2/gamma/read:0
(batch_normalization_2/moving_mean/read:0
,batch_normalization_2/moving_variance/read:0
conv2d_2/Relu:0V
!batch_normalization_2/beta/read:01batch_normalization_2/cond/batchnorm/sub/Switch:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0W
"batch_normalization_2/gamma/read:01batch_normalization_2/cond/batchnorm/mul/Switch:0F
conv2d_2/Relu:03batch_normalization_2/cond/batchnorm/mul_1/Switch:0_
(batch_normalization_2/moving_mean/read:03batch_normalization_2/cond/batchnorm/mul_2/Switch:0a
,batch_normalization_2/moving_variance/read:01batch_normalization_2/cond/batchnorm/add/Switch:0N
%batch_normalization_2/cond/switch_f:0%batch_normalization_2/cond/switch_f:0
ђ
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *і
'batch_normalization_3/batchnorm/add_1:0
%batch_normalization_3/cond/Switch_1:0
%batch_normalization_3/cond/Switch_1:1
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0N
%batch_normalization_3/cond/switch_t:0%batch_normalization_3/cond/switch_t:0P
'batch_normalization_3/batchnorm/add_1:0%batch_normalization_3/cond/Switch_1:1
љ
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*≈
!batch_normalization_3/beta/read:0
,batch_normalization_3/cond/batchnorm/Rsqrt:0
1batch_normalization_3/cond/batchnorm/add/Switch:0
,batch_normalization_3/cond/batchnorm/add/y:0
*batch_normalization_3/cond/batchnorm/add:0
,batch_normalization_3/cond/batchnorm/add_1:0
1batch_normalization_3/cond/batchnorm/mul/Switch:0
*batch_normalization_3/cond/batchnorm/mul:0
3batch_normalization_3/cond/batchnorm/mul_1/Switch:0
,batch_normalization_3/cond/batchnorm/mul_1:0
3batch_normalization_3/cond/batchnorm/mul_2/Switch:0
,batch_normalization_3/cond/batchnorm/mul_2:0
1batch_normalization_3/cond/batchnorm/sub/Switch:0
*batch_normalization_3/cond/batchnorm/sub:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
"batch_normalization_3/gamma/read:0
(batch_normalization_3/moving_mean/read:0
,batch_normalization_3/moving_variance/read:0
conv2d_3/Relu:0_
(batch_normalization_3/moving_mean/read:03batch_normalization_3/cond/batchnorm/mul_2/Switch:0a
,batch_normalization_3/moving_variance/read:01batch_normalization_3/cond/batchnorm/add/Switch:0W
"batch_normalization_3/gamma/read:01batch_normalization_3/cond/batchnorm/mul/Switch:0N
%batch_normalization_3/cond/switch_f:0%batch_normalization_3/cond/switch_f:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0V
!batch_normalization_3/beta/read:01batch_normalization_3/cond/batchnorm/sub/Switch:0F
conv2d_3/Relu:03batch_normalization_3/cond/batchnorm/mul_1/Switch:0
ђ
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *і
'batch_normalization_4/batchnorm/add_1:0
%batch_normalization_4/cond/Switch_1:0
%batch_normalization_4/cond/Switch_1:1
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0N
%batch_normalization_4/cond/switch_t:0%batch_normalization_4/cond/switch_t:0P
'batch_normalization_4/batchnorm/add_1:0%batch_normalization_4/cond/Switch_1:1L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0
љ
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*≈
!batch_normalization_4/beta/read:0
,batch_normalization_4/cond/batchnorm/Rsqrt:0
1batch_normalization_4/cond/batchnorm/add/Switch:0
,batch_normalization_4/cond/batchnorm/add/y:0
*batch_normalization_4/cond/batchnorm/add:0
,batch_normalization_4/cond/batchnorm/add_1:0
1batch_normalization_4/cond/batchnorm/mul/Switch:0
*batch_normalization_4/cond/batchnorm/mul:0
3batch_normalization_4/cond/batchnorm/mul_1/Switch:0
,batch_normalization_4/cond/batchnorm/mul_1:0
3batch_normalization_4/cond/batchnorm/mul_2/Switch:0
,batch_normalization_4/cond/batchnorm/mul_2:0
1batch_normalization_4/cond/batchnorm/sub/Switch:0
*batch_normalization_4/cond/batchnorm/sub:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
"batch_normalization_4/gamma/read:0
(batch_normalization_4/moving_mean/read:0
,batch_normalization_4/moving_variance/read:0
conv2d_4/Relu:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0N
%batch_normalization_4/cond/switch_f:0%batch_normalization_4/cond/switch_f:0W
"batch_normalization_4/gamma/read:01batch_normalization_4/cond/batchnorm/mul/Switch:0F
conv2d_4/Relu:03batch_normalization_4/cond/batchnorm/mul_1/Switch:0V
!batch_normalization_4/beta/read:01batch_normalization_4/cond/batchnorm/sub/Switch:0_
(batch_normalization_4/moving_mean/read:03batch_normalization_4/cond/batchnorm/mul_2/Switch:0a
,batch_normalization_4/moving_variance/read:01batch_normalization_4/cond/batchnorm/add/Switch:0
ђ
$batch_normalization_5/cond/cond_text$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_t:0 *і
'batch_normalization_5/batchnorm/add_1:0
%batch_normalization_5/cond/Switch_1:0
%batch_normalization_5/cond/Switch_1:1
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_t:0N
%batch_normalization_5/cond/switch_t:0%batch_normalization_5/cond/switch_t:0P
'batch_normalization_5/batchnorm/add_1:0%batch_normalization_5/cond/Switch_1:1L
$batch_normalization_5/cond/pred_id:0$batch_normalization_5/cond/pred_id:0
љ
&batch_normalization_5/cond/cond_text_1$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_f:0*≈
!batch_normalization_5/beta/read:0
,batch_normalization_5/cond/batchnorm/Rsqrt:0
1batch_normalization_5/cond/batchnorm/add/Switch:0
,batch_normalization_5/cond/batchnorm/add/y:0
*batch_normalization_5/cond/batchnorm/add:0
,batch_normalization_5/cond/batchnorm/add_1:0
1batch_normalization_5/cond/batchnorm/mul/Switch:0
*batch_normalization_5/cond/batchnorm/mul:0
3batch_normalization_5/cond/batchnorm/mul_1/Switch:0
,batch_normalization_5/cond/batchnorm/mul_1:0
3batch_normalization_5/cond/batchnorm/mul_2/Switch:0
,batch_normalization_5/cond/batchnorm/mul_2:0
1batch_normalization_5/cond/batchnorm/sub/Switch:0
*batch_normalization_5/cond/batchnorm/sub:0
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_f:0
"batch_normalization_5/gamma/read:0
(batch_normalization_5/moving_mean/read:0
,batch_normalization_5/moving_variance/read:0
conv2d_5/Relu:0L
$batch_normalization_5/cond/pred_id:0$b