function e(e,t){for(var n=0;n<t.length;n++){const r=t[n];if("string"!=typeof r&&!Array.isArray(r))for(const t in r)if("default"!==t&&!(t in e)){const n=Object.getOwnPropertyDescriptor(r,t);n&&Object.defineProperty(e,t,n.get?n:{enumerable:!0,get:()=>r[t]})}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class t{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class n{refCount(e){return r("refCount")}incRef(e){return r("incRef")}timerAvailable(){return!0}time(e){return r("time")}read(e){return r("read")}readSync(e){return r("readSync")}readToGPU(e,t){return r("readToGPU")}numDataIds(){return r("numDataIds")}disposeData(e,t){return r("disposeData")}write(e,t,n){return r("write")}move(e,t,n,s,a){return r("move")}createTensorFromGPUData(e,t,n){return r("createTensorFromGPUData")}memory(){return r("memory")}floatPrecision(){return r("floatPrecision")}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}dispose(){return r("dispose")}}function r(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function s(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,i(e,t,n)}function a(e,t,n){return Math.max(e,Math.min(t,n))}function o(e){return e%2==0?e:e+1}function i(e,t,n){const r=e[t];e[t]=e[n],e[n]=r}function l(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function u(e,t){if(!e)throw new Error("string"==typeof t?t:t())}function c(e,t,n=""){u(g(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function h(e){u(null!=e,()=>"The input to the tensor constructor must be a non-null value.")}function d(e){if(0===e.length)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function f(e){return 0===e.length}function p(e,t){if(e===t)return!0;if(null==e||null==t)return!1;if(e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(null!==e[n]&&null!==t[n]&&e[n]!==t[n])return!1;return!0}function g(e,t){if(e===t)return!0;if(null==e||null==t)return!1;if(e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function m(e){return e%1==0}function b(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function y(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function w(e,t=e=>0,n,r){return new Promise((s,a)=>{let o=0;const i=()=>{if(e())return void s();o++;const l=t(o);null!=n&&o>=n?a():null!=r?r(i,l):setTimeout(i,l)};i()})}function v(e,t){let n=1,r=-1;for(let a=0;a<e.length;++a)if(e[a]>=0)n*=e[a];else if(-1===e[a]){if(-1!==r)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${a}`);r=a}else if(e[a]<0)throw Error(`Shapes can not be < 0. Found ${e[a]} at dim ${a}`);if(-1===r){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(0===n)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const s=e.slice();return s[r]=t/n,s}function k(e,t){const n=t.length;return u((e=null==e?t.map((e,t)=>t):[].concat(e)).every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),u(e.every(e=>m(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function x(e,t){const n=[],r=[],s=null!=t&&Array.isArray(t)&&0===t.length,a=null==t||s?null:k(t,e).sort();let o=0;for(let i=0;i<e.length;++i){if(null!=a){if(a[o]===i&&1!==e[i])throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(null==a[o]||a[o]>i)&&1===e[i]&&(n.push(e[i]),r.push(i)),a[o]<=i&&o++}1!==e[i]&&(n.push(e[i]),r.push(i))}return{newShape:n,keptDims:r}}function E(e,t){return S(e,t)}function S(e,t){let n=null;if(null==e||"float32"===e)n=new Float32Array(t);else if("int32"===e)n=new Int32Array(t);else if("bool"===e)n=new Uint8Array(t);else{if("string"!==e)throw new Error(`Unknown data type ${e}`);n=new Array(t)}return n}function $(e,t){for(let n=0;n<e.length;n++){const r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function A(e){return"bool"===e||"complex64"===e||"float32"===e||"int32"===e||"string"===e}function _(e,t){return"complex64"!==t&&(("float32"!==t||"complex64"===e)&&(("int32"!==t||"float32"===e||"complex64"===e)&&("bool"!==t||"bool"!==e)))}function I(e){if("float32"===e||"int32"===e)return 4;if("complex64"===e)return 8;if("bool"===e)return 1;throw new Error(`Unknown dtype ${e}`)}function M(e){if(null==e)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function T(e){return"string"==typeof e||e instanceof String}function N(e){return"boolean"==typeof e}function D(e){return"number"==typeof e}function R(e){return Array.isArray(e)?R(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":D(e)?"float32":T(e)?"string":N(e)?"bool":"float32"}function B(e){return!!(e&&e.constructor&&e.call&&e.apply)}function F(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function C(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function O(e,t,n,r=!1){const s=new Array;if(1===t.length){const a=t[0]*(r?2:1);for(let t=0;t<a;t++)s[t]=n[e+t]}else{const a=t[0],o=t.slice(1),i=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)s[t]=O(e+t*i,o,n,r)}return s}function P(e,t,n=!1){if(0===e.length)return t[0];const r=e.reduce((e,t)=>e*t)*(n?2:1);if(0===r)return[];if(r!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return O(0,e,t,n)}function z(e,t){if(Array.isArray(e))return e;if("float32"===t)return e instanceof Float32Array?e:new Float32Array(e);if("int32"===t)return e instanceof Int32Array?e:new Int32Array(e);if("bool"===t||"string"===t)return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function L(e,t){const n=U(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function U(e,t){if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t)return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function W(e){e.forEach(t=>{u(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function G(e,t,n){if(0===t)return 0;if(1===t)return e[0];let r=e[e.length-1];for(let s=0;s<e.length-1;++s)r+=n[s]*e[s];return r}function K(e,t,n){if(0===t)return[];if(1===t)return[e];const r=new Array(t);for(let s=0;s<r.length-1;++s)r[s]=Math.floor(e/n[s]),e-=r[s]*n[s];return r[r.length-1]=e,r}function q(e){return e&&e.then&&"function"==typeof e.then}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const j="tfjsflags";class V{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=H,this.populateURLFlags()}setPlatform(e,t){null!=this.platform&&(J().getBool("IS_TEST")||J().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},null!=this.urlFlags[e]){const t=this.urlFlags[e];J().getBool("IS_TEST")||J().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(q(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(null==this.flagRegistry[e])throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,null!=this.flagRegistry[e].setHook&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(null==this.flagRegistry[e])throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(void 0===this.global||void 0===this.global.location||void 0===this.global.location.search)return;const e=this.getQueryParams(this.global.location.search);if(j in e){e[j].split(",").forEach(e=>{const[t,n]=e.split(":");this.urlFlags[t]=function(e,t){const n=t.toLowerCase();return"true"===n||"false"===n?"true"===n:""+ +n===n?+n:t}(0,n)})}}}function H(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(function(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}(t,n[0],n[1]),n.join("="))),t}function J(){return Y}let Z,Y=null;function X(){if(null==Z){let e;if("undefined"!=typeof window)e=window;else if("undefined"!=typeof global)e=global;else if("undefined"!=typeof process)e=process;else{if("undefined"==typeof self)throw new Error("Could not find a global object");e=self}Z=e}return Z}function Q(e,t){const n=function(){const e=X();return null==e._tfGlobals&&(e._tfGlobals=new Map),e._tfGlobals}();if(n.has(e))return n.get(e);{const r=t();return n.set(e,r),n.get(e)}}const ee="Abs",te="Acos",ne="Acosh",re="Add",se="AddN",ae="All",oe="Any",ie="ArgMax",le="ArgMin",ue="Asin",ce="Asinh",he="Atan",de="Atanh",fe="Atan2",pe="AvgPool",ge="AvgPoolGrad",me="AvgPool3D",be="AvgPool3DGrad",ye="BatchMatMul",we="BatchToSpaceND",ve="Bincount",ke="BitwiseAnd",xe="BroadcastArgs",Ee="Cast",Se="Ceil",$e="ClipByValue",Ae="Complex",_e="ComplexAbs",Ie="Concat",Me="Conv2D",Te="Conv2DBackpropFilter",Ne="Conv2DBackpropInput",De="Conv3D",Re="Conv3DBackpropFilterV2",Be="Conv3DBackpropInputV2",Fe="Cos",Ce="Cosh",Oe="Cumprod",Pe="Cumsum",ze="CropAndResize",Le="DenseBincount",Ue="DepthToSpace",We="DepthwiseConv2dNative",Ge="DepthwiseConv2dNativeBackpropFilter",Ke="DepthwiseConv2dNativeBackpropInput",qe="Diag",je="Dilation2D",Ve="Dilation2DBackpropInput",He="Dilation2DBackpropFilter",Je="Draw",Ze="RealDiv",Ye="Einsum",Xe="Elu",Qe="EluGrad",et="Erf",tt="Equal",nt="Exp",rt="ExpandDims",st="Expm1",at="FFT",ot="Fill",it="FlipLeftRight",lt="Floor",ut="FloorDiv",ct="FusedBatchNorm",ht="GatherV2",dt="GatherNd",ft="Greater",pt="GreaterEqual",gt="Identity",mt="IFFT",bt="Imag",yt="IsFinite",wt="IsInf",vt="IsNan",kt="LeakyRelu",xt="Less",Et="LessEqual",St="LinSpace",$t="Log",At="Log1p",_t="LogicalAnd",It="LogicalNot",Mt="LogicalOr",Tt="LRN",Nt="LRNGrad",Dt="Max",Rt="Maximum",Bt="MaxPool",Ft="MaxPoolGrad",Ct="MaxPool3D",Ot="MaxPool3DGrad",Pt="MaxPoolWithArgmax",zt="Mean",Lt="Min",Ut="Minimum",Wt="MirrorPad",Gt="Mod",Kt="Multinomial",qt="Multiply",jt="Neg",Vt="NotEqual",Ht="NonMaxSuppressionV3",Jt="NonMaxSuppressionV4",Zt="NonMaxSuppressionV5",Yt="OnesLike",Xt="OneHot",Qt="Pack",en="PadV2",tn="Pow",nn="Prelu",rn="Prod",sn="RaggedGather",an="RaggedRange",on="RaggedTensorToTensor",ln="Range",un="Real",cn="Reciprocal",hn="Relu",dn="Reshape",fn="ResizeNearestNeighbor",pn="ResizeNearestNeighborGrad",gn="ResizeBilinear",mn="ResizeBilinearGrad",bn="Relu6",yn="Reverse",wn="Round",vn="Rsqrt",kn="ScatterNd",xn="TensorScatterUpdate",En="SearchSorted",Sn="Select",$n="Selu",An="Slice",_n="Sin",In="Sinh",Mn="Sign",Tn="Sigmoid",Nn="Softplus",Dn="Sqrt",Rn="Sum",Bn="SpaceToBatchND",Fn="SplitV",Cn="Softmax",On="SparseFillEmptyRows",Pn="SparseReshape",zn="SparseSegmentMean",Ln="SparseSegmentSum",Un="SparseToDense",Wn="SquaredDifference",Gn="Square",Kn="StaticRegexReplace",qn="StridedSlice",jn="StringNGrams",Vn="StringSplit",Hn="StringToHashBucketFast",Jn="Sub",Zn="Tan",Yn="Tanh",Xn="Tile",Qn="TopK",er="Transform",tr="Transpose",nr="Unique",rr="Unpack",sr="UnsortedSegmentSum",ar="ZerosLike",or="Step",ir="FromPixels",lr="RotateWithOffset",ur="_FusedMatMul",cr="FusedConv2D",hr="FusedDepthwiseConv2D";
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function dr(...e){J().getBool("IS_TEST")||J().getBool("PROD")||console.warn(...e)}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const fr=Q("kernelRegistry",()=>new Map),pr=Q("gradRegistry",()=>new Map);function gr(e,t){const n=wr(e,t);return fr.get(n)}function mr(e){return pr.get(e)}function br(e){const t=fr.entries(),n=[];for(;;){const{done:r,value:s}=t.next();if(r)break;const[a,o]=s,[i]=a.split("_");i===e&&n.push(o)}return n}function yr(e){const{kernelName:t,backendName:n}=e,r=wr(t,n);fr.has(r)&&dr(`The kernel '${t}' for backend '${n}' is already registered`),fr.set(r,e)}function wr(e,t){return`${t}_${e}`}
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vr(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var kr="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof window?window:"undefined"!=typeof global?global:"undefined"!=typeof self?self:{};function xr(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function Er(e){if(e.__esModule)return e;var t=e.default;if("function"==typeof t){var n=function e(){return this instanceof e?Reflect.construct(t,arguments,this.constructor):t.apply(this,arguments)};n.prototype=t.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(e).forEach(function(t){var r=Object.getOwnPropertyDescriptor(e,t);Object.defineProperty(n,t,r.get?r:{enumerable:!0,get:function(){return e[t]}})}),n}var Sr=Ar,$r=null;try{$r=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch(Lp){}function Ar(e,t,n){this.low=0|e,this.high=0|t,this.unsigned=!!n}function _r(e){return!0===(e&&e.__isLong__)}Ar.prototype.__isLong__,Object.defineProperty(Ar.prototype,"__isLong__",{value:!0}),Ar.isLong=_r;var Ir={},Mr={};function Tr(e,t){var n,r,s;return t?(s=0<=(e>>>=0)&&e<256)&&(r=Mr[e])?r:(n=Dr(e,(0|e)<0?-1:0,!0),s&&(Mr[e]=n),n):(s=-128<=(e|=0)&&e<128)&&(r=Ir[e])?r:(n=Dr(e,e<0?-1:0,!1),s&&(Ir[e]=n),n)}function Nr(e,t){if(isNaN(e))return t?Ur:Lr;if(t){if(e<0)return Ur;if(e>=Or)return jr}else{if(e<=-Pr)return Vr;if(e+1>=Pr)return qr}return e<0?Nr(-e,t).neg():Dr(e%Cr|0,e/Cr|0,t)}function Dr(e,t,n){return new Ar(e,t,n)}Ar.fromInt=Tr,Ar.fromNumber=Nr,Ar.fromBits=Dr;var Rr=Math.pow;function Br(e,t,n){if(0===e.length)throw Error("empty string");if("NaN"===e||"Infinity"===e||"+Infinity"===e||"-Infinity"===e)return Lr;if("number"==typeof t?(n=t,t=!1):t=!!t,(n=n||10)<2||36<n)throw RangeError("radix");var r;if((r=e.indexOf("-"))>0)throw Error("interior hyphen");if(0===r)return Br(e.substring(1),t,n).neg();for(var s=Nr(Rr(n,8)),a=Lr,o=0;o<e.length;o+=8){var i=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+i),n);if(i<8){var u=Nr(Rr(n,i));a=a.mul(u).add(Nr(l))}else a=(a=a.mul(s)).add(Nr(l))}return a.unsigned=t,a}function Fr(e,t){return"number"==typeof e?Nr(e,t):"string"==typeof e?Br(e,t):Dr(e.low,e.high,"boolean"==typeof t?t:e.unsigned)}Ar.fromString=Br,Ar.fromValue=Fr;var Cr=4294967296,Or=Cr*Cr,Pr=Or/2,zr=Tr(1<<24),Lr=Tr(0);Ar.ZERO=Lr;var Ur=Tr(0,!0);Ar.UZERO=Ur;var Wr=Tr(1);Ar.ONE=Wr;var Gr=Tr(1,!0);Ar.UONE=Gr;var Kr=Tr(-1);Ar.NEG_ONE=Kr;var qr=Dr(-1,2147483647,!1);Ar.MAX_VALUE=qr;var jr=Dr(-1,-1,!0);Ar.MAX_UNSIGNED_VALUE=jr;var Vr=Dr(0,-2147483648,!1);Ar.MIN_VALUE=Vr;var Hr=Ar.prototype;Hr.toInt=function(){return this.unsigned?this.low>>>0:this.low},Hr.toNumber=function(){return this.unsigned?(this.high>>>0)*Cr+(this.low>>>0):this.high*Cr+(this.low>>>0)},Hr.toString=function(e){if((e=e||10)<2||36<e)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative()){if(this.eq(Vr)){var t=Nr(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return"-"+this.neg().toString(e)}for(var s=Nr(Rr(e,6),this.unsigned),a=this,o="";;){var i=a.div(s),l=(a.sub(i.mul(s)).toInt()>>>0).toString(e);if((a=i).isZero())return l+o;for(;l.length<6;)l="0"+l;o=""+l+o}},Hr.getHighBits=function(){return this.high},Hr.getHighBitsUnsigned=function(){return this.high>>>0},Hr.getLowBits=function(){return this.low},Hr.getLowBitsUnsigned=function(){return this.low>>>0},Hr.getNumBitsAbs=function(){if(this.isNegative())return this.eq(Vr)?64:this.neg().getNumBitsAbs();for(var e=0!=this.high?this.high:this.low,t=31;t>0&&!(e&1<<t);t--);return 0!=this.high?t+33:t+1},Hr.isZero=function(){return 0===this.high&&0===this.low},Hr.eqz=Hr.isZero,Hr.isNegative=function(){return!this.unsigned&&this.high<0},Hr.isPositive=function(){return this.unsigned||this.high>=0},Hr.isOdd=function(){return!(1&~this.low)},Hr.isEven=function(){return!(1&this.low)},Hr.equals=function(e){return _r(e)||(e=Fr(e)),(this.unsigned===e.unsigned||this.high>>>31!=1||e.high>>>31!=1)&&(this.high===e.high&&this.low===e.low)},Hr.eq=Hr.equals,Hr.notEquals=function(e){return!this.eq(e)},Hr.neq=Hr.notEquals,Hr.ne=Hr.notEquals,Hr.lessThan=function(e){return this.comp(e)<0},Hr.lt=Hr.lessThan,Hr.lessThanOrEqual=function(e){return this.comp(e)<=0},Hr.lte=Hr.lessThanOrEqual,Hr.le=Hr.lessThanOrEqual,Hr.greaterThan=function(e){return this.comp(e)>0},Hr.gt=Hr.greaterThan,Hr.greaterThanOrEqual=function(e){return this.comp(e)>=0},Hr.gte=Hr.greaterThanOrEqual,Hr.ge=Hr.greaterThanOrEqual,Hr.compare=function(e){if(_r(e)||(e=Fr(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},Hr.comp=Hr.compare,Hr.negate=function(){return!this.unsigned&&this.eq(Vr)?Vr:this.not().add(Wr)},Hr.neg=Hr.negate,Hr.add=function(e){_r(e)||(e=Fr(e));var t=this.high>>>16,n=65535&this.high,r=this.low>>>16,s=65535&this.low,a=e.high>>>16,o=65535&e.high,i=e.low>>>16,l=0,u=0,c=0,h=0;return c+=(h+=s+(65535&e.low))>>>16,u+=(c+=r+i)>>>16,l+=(u+=n+o)>>>16,l+=t+a,Dr((c&=65535)<<16|(h&=65535),(l&=65535)<<16|(u&=65535),this.unsigned)},Hr.subtract=function(e){return _r(e)||(e=Fr(e)),this.add(e.neg())},Hr.sub=Hr.subtract,Hr.multiply=function(e){if(this.isZero())return Lr;if(_r(e)||(e=Fr(e)),$r)return Dr($r.mul(this.low,this.high,e.low,e.high),$r.get_high(),this.unsigned);if(e.isZero())return Lr;if(this.eq(Vr))return e.isOdd()?Vr:Lr;if(e.eq(Vr))return this.isOdd()?Vr:Lr;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(zr)&&e.lt(zr))return Nr(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,n=65535&this.high,r=this.low>>>16,s=65535&this.low,a=e.high>>>16,o=65535&e.high,i=e.low>>>16,l=65535&e.low,u=0,c=0,h=0,d=0;return h+=(d+=s*l)>>>16,c+=(h+=r*l)>>>16,h&=65535,c+=(h+=s*i)>>>16,u+=(c+=n*l)>>>16,c&=65535,u+=(c+=r*i)>>>16,c&=65535,u+=(c+=s*o)>>>16,u+=t*l+n*i+r*o+s*a,Dr((h&=65535)<<16|(d&=65535),(u&=65535)<<16|(c&=65535),this.unsigned)},Hr.mul=Hr.multiply,Hr.divide=function(e){if(_r(e)||(e=Fr(e)),e.isZero())throw Error("division by zero");var t,n,r;if($r)return this.unsigned||-2147483648!==this.high||-1!==e.low||-1!==e.high?Dr((this.unsigned?$r.div_u:$r.div_s)(this.low,this.high,e.low,e.high),$r.get_high(),this.unsigned):this;if(this.isZero())return this.unsigned?Ur:Lr;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return Ur;if(e.gt(this.shru(1)))return Gr;r=Ur}else{if(this.eq(Vr))return e.eq(Wr)||e.eq(Kr)?Vr:e.eq(Vr)?Wr:(t=this.shr(1).div(e).shl(1)).eq(Lr)?e.isNegative()?Wr:Kr:(n=this.sub(e.mul(t)),r=t.add(n.div(e)));if(e.eq(Vr))return this.unsigned?Ur:Lr;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();r=Lr}for(n=this;n.gte(e);){t=Math.max(1,Math.floor(n.toNumber()/e.toNumber()));for(var s=Math.ceil(Math.log(t)/Math.LN2),a=s<=48?1:Rr(2,s-48),o=Nr(t),i=o.mul(e);i.isNegative()||i.gt(n);)i=(o=Nr(t-=a,this.unsigned)).mul(e);o.isZero()&&(o=Wr),r=r.add(o),n=n.sub(i)}return r},Hr.div=Hr.divide,Hr.modulo=function(e){return _r(e)||(e=Fr(e)),$r?Dr((this.unsigned?$r.rem_u:$r.rem_s)(this.low,this.high,e.low,e.high),$r.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},Hr.mod=Hr.modulo,Hr.rem=Hr.modulo,Hr.not=function(){return Dr(~this.low,~this.high,this.unsigned)},Hr.and=function(e){return _r(e)||(e=Fr(e)),Dr(this.low&e.low,this.high&e.high,this.unsigned)},Hr.or=function(e){return _r(e)||(e=Fr(e)),Dr(this.low|e.low,this.high|e.high,this.unsigned)},Hr.xor=function(e){return _r(e)||(e=Fr(e)),Dr(this.low^e.low,this.high^e.high,this.unsigned)},Hr.shiftLeft=function(e){return _r(e)&&(e=e.toInt()),0==(e&=63)?this:e<32?Dr(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):Dr(0,this.low<<e-32,this.unsigned)},Hr.shl=Hr.shiftLeft,Hr.shiftRight=function(e){return _r(e)&&(e=e.toInt()),0==(e&=63)?this:e<32?Dr(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):Dr(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},Hr.shr=Hr.shiftRight,Hr.shiftRightUnsigned=function(e){if(_r(e)&&(e=e.toInt()),0===(e&=63))return this;var t=this.high;return e<32?Dr(this.low>>>e|t<<32-e,t>>>e,this.unsigned):Dr(32===e?t:t>>>e-32,0,this.unsigned)},Hr.shru=Hr.shiftRightUnsigned,Hr.shr_u=Hr.shiftRightUnsigned,Hr.toSigned=function(){return this.unsigned?Dr(this.low,this.high,!1):this},Hr.toUnsigned=function(){return this.unsigned?this:Dr(this.low,this.high,!0)},Hr.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},Hr.toBytesLE=function(){var e=this.high,t=this.low;return[255&t,t>>>8&255,t>>>16&255,t>>>24,255&e,e>>>8&255,e>>>16&255,e>>>24]},Hr.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,255&e,t>>>24,t>>>16&255,t>>>8&255,255&t]},Ar.fromBytes=function(e,t,n){return n?Ar.fromBytesLE(e,t):Ar.fromBytesBE(e,t)},Ar.fromBytesLE=function(e,t){return new Ar(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},Ar.fromBytesBE=function(e,t){return new Ar(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)};const Jr=xr(Sr),Zr=Jr||e({__proto__:null,default:Jr},[Sr]);function Yr(e){return Zr.fromString(e,!0,16)}const Xr=Yr("c3a5c85c97cb3127"),Qr=Yr("b492b66fbe98f273"),es=Yr("9ae16a3b2f90404f");function ts(e){return e.xor(e.shru(47))}function ns(e,t,n){const r=e.slice(t,t+n);return Zr.fromBytes(Array.from(r),!0,!0)}function rs(e,t){return ns(e,t,8)}function ss(e,t){return ns(e,t,4)}function as(e,t){return 0===t?e:e.shru(t).or(e.shl(64-t))}function os(e,t,n=Yr("9ddfea08eb382d69")){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let s=t.xor(r).mul(n);return s=s.xor(s.shru(47)),s=s.mul(n),s}function is(e,t,n,r){return function(e,t,n,r,s,a){s=s.add(e),a=as(a.add(s).add(r),21);const o=s;return s=(s=s.add(t)).add(n),a=a.add(as(s,44)),[s.add(r),a.add(o)]}(rs(e,t),rs(e,t+8),rs(e,t+16),rs(e,t+24),n,r)}function ls(e,t=e.length){const n=Zr.fromNumber(81,!0);if(t<=32)return t<=16?function(e,t=e.length){if(t>=8){const n=es.add(2*t),r=rs(e,0).add(es),s=rs(e,t-8);return os(as(s,37).mul(n).add(r),as(r,25).add(s).mul(n),n)}if(t>=4){const n=es.add(2*t);return os(ss(e,0).shl(3).add(t),ss(e,t-4),n)}if(t>0){const n=e[0]+(e[t>>1]<<8),r=t+(e[t-1]<<2);return ts(es.mul(n).xor(Xr.mul(r))).mul(es)}return es}(e,t):function(e,t=e.length){const n=es.add(2*t),r=rs(e,0).mul(Qr),s=rs(e,8),a=rs(e,t-8).mul(n),o=rs(e,t-16).mul(es);return os(as(r.add(s),43).add(as(a,30)).add(o),r.add(as(s.add(es),18)).add(a),n)}(e,t);if(t<=64)return function(e,t=e.length){const n=es.add(2*t),r=rs(e,0).mul(es),s=rs(e,8),a=rs(e,t-8).mul(n),o=rs(e,t-16).mul(es),i=as(r.add(s),43).add(as(a,30)).add(o),l=os(i,r.add(as(s.add(es),18)).add(a),n),u=rs(e,16).mul(n),c=rs(e,24),h=i.add(rs(e,t-32)).mul(n),d=l.add(rs(e,t-24)).mul(n);return os(as(u.add(c),43).add(as(h,30)).add(d),u.add(as(c.add(r),18)).add(h),n)}(e,t);let r=n,s=n.mul(Qr).add(113),a=ts(s.mul(es).add(113)).mul(es),o=[Zr.UZERO,Zr.UZERO],i=[Zr.UZERO,Zr.UZERO];r=r.mul(es).add(rs(e,0));let l=0;const u=64*(t-1>>6),c=u+(t-1&63)-63;do{r=as(r.add(s).add(o[0]).add(rs(e,l+8)),37).mul(Qr),s=as(s.add(o[1]).add(rs(e,l+48)),42).mul(Qr),r=r.xor(i[1]),s=s.add(o[0]).add(rs(e,l+40)),a=as(a.add(i[0]),33).mul(Qr),o=is(e,l,o[1].mul(Qr),r.add(i[0])),i=is(e,l+32,a.add(i[1]),s.add(rs(e,l+16))),[a,r]=[r,a],l+=64}while(l!==u);const h=Qr.add(a.and(255).shl(1));return l=c,i[0]=i[0].add(t-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),r=as(r.add(s).add(o[0]).add(rs(e,l+8)),37).mul(h),s=as(s.add(o[1]).add(rs(e,l+48)),42).mul(h),r=r.xor(i[1].mul(9)),s=s.add(o[0].mul(9).add(rs(e,l+40))),a=as(a.add(i[0]),33).mul(h),o=is(e,l,o[1].mul(h),r.add(i[0])),i=is(e,l+32,a.add(i[1]),s.add(rs(e,l+16))),[a,r]=[r,a],os(os(o[0],i[0],h).add(ts(s).mul(Xr)).add(a),os(o[1],i[1],h).add(r),h)}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function us(e,t){return"string"===t?ds(e):cs([e],t)}function cs(e,t){if("string"===t)throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=gs(e)),J().getBool("DEBUG")&&$(e,t),function(e,t){return e instanceof Float32Array&&"float32"===t||e instanceof Int32Array&&"int32"===t||e instanceof Uint8Array&&"bool"===t}(e,t))return e;if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t){const t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)0!==Math.round(e[n])&&(t[n]=1);return t}throw new Error(`Unknown data type ${t}`)}function hs(){return J().platform.now()}function ds(e,t="utf-8"){return t=t||"utf-8",J().platform.encode(e,t)}function fs(e,t="utf-8"){return t=t||"utf-8",J().platform.decode(e,t)}function ps(e){return null!=J().platform.isTypedArray?J().platform.isTypedArray(e):vr(e)}function gs(e,t=[],n=!1){if(null==t&&(t=[]),"boolean"==typeof e||"number"==typeof e||"string"==typeof e||q(e)||null==e||ps(e)&&n)t.push(e);else if(Array.isArray(e)||ps(e))for(let r=0;r<e.length;++r)gs(e[r],t,n);else{let r=-1;for(const t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let s=0;s<=r;s++)gs(e[s],t,n)}return t}const ms=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:g,arraysEqualWithNull:p,assert:u,assertNonNegativeIntegerDimensions:W,assertNonNull:h,assertShapesMatch:c,bytesFromStringArray:M,bytesPerElement:I,checkConversionForErrors:$,clamp:a,computeStrides:C,convertBackendValuesAndArrayBuffer:z,createScalarValue:us,createShuffledIndices:function(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return s(t),t},decodeString:fs,distSquared:function(e,t){let n=0;for(let r=0;r<e.length;r++){const s=Number(e[r])-Number(t[r]);n+=s*s}return n},encodeString:ds,fetch:function(e,t){return J().platform.fetch(e,t)},fingerPrint64:ls,flatten:gs,getArrayFromDType:S,getTypedArrayFromDType:E,hasEncodingLoss:_,hexToLong:Yr,indexToLoc:K,inferDtype:R,inferFromImplicitShape:v,isBoolean:N,isFunction:B,isInt:m,isNumber:D,isPromise:q,isScalarShape:f,isString:T,isTypedArray:ps,isValidDtype:A,locToIndex:G,makeOnesTypedArray:L,makeZerosNestedTypedArray:function(e,t){const n=e.reduce((e,t)=>e*t,1);if(null==t||"float32"===t)return P(e,new Float32Array(n));if("int32"===t)return P(e,new Int32Array(n));if("bool"===t)return P(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)},makeZerosTypedArray:U,nearestDivisor:F,nearestLargerEven:o,now:hs,parseAxisParam:k,randUniform:function(e,t){const n=Math.random();return t*n+(1-n)*e},repeatedTry:w,rightPad:y,shuffle:s,shuffleCombo:function(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,n--,i(e,n,r),i(t,n,r)},sizeFromShape:d,sizeToSquarishShape:b,squeezeShape:x,sum:l,swap:i,tanh:function(e){if(null!=Math.tanh)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}},toNestedArray:P,toTypedArray:cs},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bs{constructor(e,t){this.backendTimer=e,this.logger=t,null==t&&(this.logger=new ws)}profileKernel(e,t,n){let r;const s=()=>{r=n()};let a;const o=hs();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(s);else{s();for(const e of r)e.dataSync();a=Promise.resolve({kernelMs:hs()-o})}if(J().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let i=0;i<r.length;i++){const t=r[i];t.data().then(n=>{ys(n,t.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>null!=e.getExtraProfileInfo?e.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:n,timeMs:r,inputs:s,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],s,n[2])})})}}function ys(e,t,n){if("float32"!==t)return!1;for(let r=0;r<e.length;r++){const t=e[r];if(isNaN(t)||!isFinite(t))return console.warn(`Found ${t} in the result of '${n}'`),!0}return!1}class ws{logKernelProfile(e,t,n,r,s,a){const o="number"==typeof r?y(`${r}ms`,9):r.error,i=y(e,25),l=t.rank,u=t.size,c=y(t.shape.toString(),14);let h="";for(const d in s){const e=s[d];if(null!=e){const n=e.shape||t.shape,r=n.length;h+=`${d}: ${r}D ${r>0?n:""} `}}console.log(`%c${i}\t%c${o}\t%c${l}D ${c}\t%c${u}\t%c${h}\t%c${a}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vs(e,t,n,r){const s=C(t),a=function(e,t,n,r){const s=d(t),a=r[r.length-1],o=new Array(a).fill(0),i=t.length,l="complex64"===n?Ss(e):e;if(i>1)for(let u=0;u<s/a;u++){const e=u*a;for(let t=0;t<a;t++)o[t]=Math.max(o[t],ks(l[e+t],0,n).length)}return o}(e,t,n,s),o=t.length,i=Es(e,t,n,s,a),l=["Tensor"];return r&&(l.push(`  dtype: ${n}`),l.push(`  rank: ${o}`),l.push(`  shape: [${t}]`),l.push("  values:")),l.push(i.map(e=>"    "+e).join("\n")),l.join("\n")}function ks(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(7))} + ${parseFloat(e[1].toFixed(7))}j`:T(e)?`'${e}'`:"bool"===n?xs(e):parseFloat(e.toFixed(7)).toString(),y(r,t)}function xs(e){return 0===e?"false":"true"}function Es(e,t,n,r,s,a=!0){const o="complex64"===n?2:1,i=t[0],l=t.length;if(0===l){if("complex64"===n){return[ks(Ss(e)[0],0,n)]}return"bool"===n?[xs(e[0])]:[e[0].toString()]}if(1===l){if(i>20){const t=3*o;let r=Array.from(e.slice(0,t)),a=Array.from(e.slice((i-3)*o,i*o));return"complex64"===n&&(r=Ss(r),a=Ss(a)),["["+r.map((e,t)=>ks(e,s[t],n)).join(", ")+", ..., "+a.map((e,t)=>ks(e,s[i-3+t],n)).join(", ")+"]"]}return["["+("complex64"===n?Ss(e):Array.from(e)).map((e,t)=>ks(e,s[t],n)).join(", ")+"]"]}const u=t.slice(1),c=r.slice(1),h=r[0]*o,d=[];if(i>20){for(let t=0;t<3;t++){const r=t*h,a=r+h;d.push(...Es(e.slice(r,a),u,n,c,s,!1))}d.push("...");for(let t=i-3;t<i;t++){const r=t*h,a=r+h;d.push(...Es(e.slice(r,a),u,n,c,s,t===i-1))}}else for(let g=0;g<i;g++){const t=g*h,r=t+h;d.push(...Es(e.slice(t,r),u,n,c,s,g===i-1))}const f=2===l?",":"";d[0]="["+(i>0?d[0]+f:"");for(let g=1;g<d.length-1;g++)d[g]=" "+d[g]+f;let p=",\n";for(let g=2;g<l;g++)p+="\n";return d[d.length-1]=" "+d[d.length-1]+"]"+(a?"":p),d}function Ss(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $s{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=d(e),null!=n){const e=n.length;u(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if("complex64"===t)throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||S(t,this.size),this.strides=C(e)}set(e,...t){0===t.length&&(t=[0]),u(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const n=this.locToIndex(t);this.values[n]=e}get(...e){0===e.length&&(e=[0]);let t=0;for(const r of e){if(r<0||r>=this.shape[t]){const t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(t)}t++}let n=e[e.length-1];for(let r=0;r<e.length-1;++r)n+=this.strides[r]*e[r];return this.values[n]}locToIndex(e){if(0===this.rank)return 0;if(1===this.rank)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(0===this.rank)return[];if(1===this.rank)return[e];const t=new Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return As().makeTensor(this.values,this.shape,this.dtype)}}let As=null,_s=null;class Is{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=d(e),this.strides=C(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return _s.buffer(this.shape,this.dtype,e)}bufferSync(){return _s.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return P(this.shape,e,"complex64"===this.dtype)}arraySync(){return P(this.shape,this.dataSync(),"complex64"===this.dtype)}async data(){this.throwIfDisposed();const e=As().read(this.dataId);if("string"===this.dtype){const n=await e;try{return n.map(e=>fs(e))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),As().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=As().readSync(this.dataId);if("string"===this.dtype)try{return e.map(e=>fs(e))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await As().read(this.dataId);return"string"===this.dtype?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),As().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return _s.print(this,e)}clone(){return this.throwIfDisposed(),_s.clone(this)}toString(e=!1){return vs(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),_s.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),As().makeVariable(this,e,t,n)}}function Ms(){return Q("Tensor",()=>Is)}Object.defineProperty(Is,Symbol.hasInstance,{value:e=>!!e&&null!=e.data&&null!=e.dataSync&&null!=e.throwIfDisposed}),Ms();class Ts extends Is{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!g(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);As().disposeTensor(this),this.dataId=e.dataId,As().incRef(this,null)}dispose(){As().disposeVariable(this),this.isDisposedInternal=!0}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var Ns,Ds,Rs,Bs,Fs,Cs,Os,Ps,zs,Ls;Object.defineProperty(Ts,Symbol.hasInstance,{value:e=>e instanceof Is&&null!=e.assign&&e.assign instanceof Function}),(Ds=Ns||(Ns={})).R0="R0",Ds.R1="R1",Ds.R2="R2",Ds.R3="R3",Ds.R4="R4",Ds.R5="R5",Ds.R6="R6",(Bs=Rs||(Rs={})).float32="float32",Bs.int32="int32",Bs.bool="int32",Bs.complex64="complex64",(Cs=Fs||(Fs={})).float32="float32",Cs.int32="int32",Cs.bool="bool",Cs.complex64="complex64",(Ps=Os||(Os={})).float32="float32",Ps.int32="float32",Ps.bool="float32",Ps.complex64="complex64",(Ls=zs||(zs={})).float32="complex64",Ls.int32="complex64",Ls.bool="complex64",Ls.complex64="complex64";const Us={float32:Os,int32:Rs,bool:Fs,complex64:zs};function Ws(e,t){if("string"===e||"string"===t){if("string"===e&&"string"===t)return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Us[e][t]}function Gs(e){return Ws(e,"int32")}function Ks(e){return null!=e&&"object"==typeof e&&"texture"in e&&e.texture instanceof WebGLTexture}function qs(e){return"undefined"!=typeof GPUBuffer&&null!=e&&"object"==typeof e&&"buffer"in e&&e.buffer instanceof GPUBuffer}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function js(e,t){if(e.dtype===t.dtype)return[e,t];const n=Ws(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function Vs(e,t){u(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Hs(e){const t=[];return Js(e,t,new Set),t}function Js(e,t,n){if(null==e)return;if(e instanceof Is)return void t.push(e);if(r=e,!Array.isArray(r)&&"object"!=typeof r)return;var r;const s=e;for(const a in s){const e=s[a];n.has(e)||(n.add(e),Js(e,t,n))}}const Zs=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:Vs,getTensorsInContainer:Hs,isTensorInList:function(e,t){return t.some(t=>t.id===e.id)},makeTypesMatch:js},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ys(e){return null!=e.kernelName}class Xs{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class Qs{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Xs}async ready(){if(null!=this.pendingBackendInit)return this.pendingBackendInit.then(()=>{});if(null!=this.backendInstance)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t];if(await this.initializeBackend(n).success)return void(await this.setBackend(n))}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(null!=this.pendingBackendInit)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(null==this.backendInstance){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(!(e in this.registryFactory))return null;{const{asyncInit:t}=this.initializeBackend(e);if(t)return null}}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(dr(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(null==this.registryFactory[e])throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,null==this.registry[e]){this.backendInstance=null;const{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new bs(this.backendInstance),!0}setupRegisteredKernels(){br(this.backendName).forEach(e=>{null!=e.setupFunc&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){br(e).forEach(t=>{null!=t.disposeFunc&&t.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(null==t)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const r=t.factory();if(!r||r instanceof n||"function"!=typeof r.then)return this.registry[e]=r,{success:!0,asyncInit:!1};{const t=++this.pendingBackendInitId,n=r.then(n=>!(t<this.pendingBackendInitId)&&(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>(t<this.pendingBackendInitId||(this.pendingBackendInit=null,dr(`Initialization of backend ${e} failed`),dr(n.stack||n.message)),!1));return this.pendingBackendInit=n,{success:n,asyncInit:!0}}}catch(r){return dr(`Initialization of backend ${e} failed`),dr(r.stack||r.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&null!=this.pendingBackendInit&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(0===Object.keys(this.registryFactory).length)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t],{success:r,asyncInit:s}=this.initializeBackend(n);if(s||r)return{name:n,asyncInit:s}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const n=this.state.tensorInfo.get(t),r=n.backend,s=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,s,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n,r=null;if(null==t){if("function"!=typeof e)throw new Error("Please provide a function to tidy()");t=e}else{if("string"!=typeof e&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if("function"!=typeof t)throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=e}return this.scopedRun(()=>this.startScope(r),()=>this.endScope(n),()=>(n=t(),n instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),n))}scopedRun(e,t,n){e();try{const e=n();return t(),e}catch(r){throw t(),r}}nextTensorId(){return Qs.nextTensorId++}nextVariableId(){return Qs.nextVariableId++}clone(e){const t=ta.runKernel(gt,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{const t={x:e},n={dtype:"float32"};return ta.runKernel(Ee,t,n)}}),[],{}),t}runKernel(e,t,n){null==this.backendName&&this.backend;if(!(null!=gr(e,this.backendName)))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,n){const r=this.backend.numDataIds();let s=0;n.forEach(e=>{s+="complex64"===e.dtype?3:1});const a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-s-a;if(o>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[];const r=this.isTapeOn(),s=this.state.numBytes,a=this.state.numTensors;let o,i;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0),null==this.backendName&&this.backend;const l=Ys(e)?e.kernelName:null!=this.state.activeScope?this.state.activeScope.name:"";if(Ys(e)){const{kernelName:t,inputs:s,attrs:a}=e;null==this.backendName&&this.backend;const l=gr(t,this.backendName);u(null!=l,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{const e=this.backend.numDataIds();i=l.kernelFunc({inputs:s,attrs:a,backend:this.backend});const o=Array.isArray(i)?i:[i];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);const u=o.map(e=>null!=e.rank?e:this.makeTensorFromTensorInfo(e));if(r){const e=this.getTensorsForGradient(t,s,u);n=this.saveTensorsForBackwardMode(e)}return u}}else{const{forwardFunc:t}=e,s=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{const e=this.backend.numDataIds();i=this.tidy(()=>t(this.backend,s));const n=Array.isArray(i)?i:[i];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,e,n),n}}const{inputs:c,attrs:h}=e,d=Ys(e)?null:e.backwardsFunc;let f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{this.ENV.getBool("DEBUG")||this.state.profiling?(f=this.profiler.profileKernel(l,c,()=>o()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(f),t=f.outputs):t=o()}),r&&this.addTapeNode(l,c,t,d,n,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-s,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(c).map(e=>null!=c[e]?c[e].shape:null),outputShapes:t.map(e=>e.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(i)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){const r=mr(e);if(null!=r){const e=r.inputsToSave||[],s=r.outputsToSave||[];let a;r.saveAllInputs?(u(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);const o=n.filter((e,t)=>s[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(null==e)throw new Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let s=e;"string"===n&&T(e[0])&&(s=e.map(e=>ds(e)));const a=r.write(s,t,n),o=new Is(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),"string"===n){const e=this.state.tensorInfo.get(a),t=M(s);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){const s={dataId:e,shape:t,dtype:n=n||"float32"};return this.makeTensorFromTensorInfo(s,r)}makeTensorFromTensorInfo(e,t){const{dataId:n,shape:r,dtype:s}=e,a=new Is(r,s,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n=n||this.nextVariableId().toString(),null!=r&&r!==e.dtype&&(e=e.cast(r));const s=new Ts(e,t,n,this.nextTensorId());if(null!=this.state.registeredVariables[s.name])throw new Error(`Variable with name ${s.name} was already registered`);return this.state.registeredVariables[s.name]=s,this.incRef(s,this.backend),s}trackTensor(e,t){this.state.numTensors++,"string"===e.dtype&&this.state.numStringTensors++;let n=0;"complex64"!==e.dtype&&"string"!==e.dtype&&(n=e.size*I(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof Ts||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,"string"===e.dtype&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),"complex64"!==e.dtype&&"string"!==e.dtype){const t=e.size*I(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),null!=this.state.registeredVariables[e.name]&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,null==e.reasons&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&0===this.state.kernelDepth}addTapeNode(e,t,n,r,s,a){const o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:s},i=mr(e);null!=i&&(r=i.gradFunc),null!=r&&(o.gradient=e=>(e=e.map((e,t)=>{if(null==e){const e=n[t],r=U(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],s,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){0===this.state.gradientDepth&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=Hs(e),n=new Set(t.map(e=>e.id));for(let s=0;s<this.state.activeScope.track.length;s++){const e=this.state.activeScope.track[s];e.kept||n.has(e.id)||e.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=0===this.state.scopeStack.length?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{e.kept||e.scopeId!==r.id||this.track(e)})}gradients(e,t,n,r=!1){if(u(t.length>0,()=>"gradients() received an empty list of xs."),null!=n&&"float32"!==n.dtype)throw new Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);const s=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));u(s instanceof Is,()=>"The result y returned by f() must be a tensor.");const a=function(e,t,n){const r={},s={};for(let l=0;l<t.length;l++)r[t[l].id]=!0;for(let l=0;l<e.length;l++){const n=e[l],a=n.inputs;for(const e in a){const o=a[e];let i=!1;for(let e=0;e<t.length;e++)if(r[o.id]){n.outputs.forEach(e=>r[e.id]=!0),i=!0,s[n.id]=!0;break}if(i)break}}const a={};a[n.id]=!0;const o={};for(let l=e.length-1;l>=0;l--){const t=e[l],n=t.inputs;for(let e=0;e<t.outputs.length;e++)if(a[t.outputs[e].id]){for(const e in n)a[n[e].id]=!0,o[t.id]=!0;break}}const i=[];for(let l=0;l<e.length;l++){const t=e[l];if(s[t.id]&&o[t.id]){const e={};for(const s in t.inputs){const n=t.inputs[s];r[n.id]&&(e[s]=n)}const n=Object.assign({},t);n.inputs=e,n.outputs=t.outputs,i.push(n)}}return i}(this.state.activeTape,t,s);if(!r&&0===a.length&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const e={};e[s.id]=null==n?function(e){const t=L(d(e),"float32");return ta.makeTensor(t,e,"float32")}(s.shape):n,function(e,t,n,r){for(let s=t.length-1;s>=0;s--){const a=t[s],o=[];if(a.outputs.forEach(t=>{const n=e[t.id];null!=n?o.push(n):o.push(null)}),null==a.gradient)throw new Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);const i=a.gradient(o);for(const t in a.inputs){if(!(t in i))throw new Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(i)}.`);const s=n(()=>i[t]());if("float32"!==s.dtype)throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${s.dtype}'`);const o=a.inputs[t];if(!g(s.shape,o.shape))throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${s.shape}', which does not match the shape of the input '${o.shape}'`);if(null==e[o.id])e[o.id]=s;else{const t=e[o.id];e[o.id]=r(t,s),t.dispose()}}}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e,a,e=>this.tidy(e),na);const r=t.map(t=>e[t.id]);return 0===this.state.gradientDepth&&(this.state.activeTape.forEach(e=>{for(const t of e.saved)t.dispose()}),this.state.activeTape=null),{value:s,grads:r}})}customGrad(e){return u(B(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{let n;u(t.every(e=>e instanceof Is),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");const r={};t.forEach((e,t)=>{r[t]=e});return this.runKernelFunc({forwardFunc:(r,s)=>(n=e(...t,s),u(n.value instanceof Is,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),u(B(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{const s=n.gradFunc(e,r),a=Array.isArray(s)?s:[s];u(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),u(a.every(e=>e instanceof Is),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=hs(),n=await this.backend.time(e);return n.wallMs=hs()-t,n}track(e){return null!=this.state.activeScope&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Xs;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}function ea(){const e=X();if(null==e._tfengine){const t=new V(e);e._tfengine=new Qs(t)}var t;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */return t=e._tfengine.ENV,Y=t,As=()=>e._tfengine,e._tfengine}Qs.nextTensorId=0,Qs.nextVariableId=0;const ta=ea();function na(e,t){const n={a:e,b:t};return ta.runKernel(re,n)}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ra;function sa(e){if(void 0!==ra)return ra;if(e||"undefined"!=typeof navigator&&null!=navigator){if(e||(e=navigator),"ReactNative"===e.product)return!0;const t=e.userAgent||e.vendor||("undefined"!=typeof window?window.opera:"");if(!t){const t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function aa(){return"undefined"!=typeof window&&null!=window.document||"undefined"!=typeof WorkerGlobalScope}const oa=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:aa,isMobile:sa,mockIsMobile:function(e){ra=e}},Symbol.toStringTag,{value:"Module"})),ia=J();
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function la(e,t){let n=e;if(ps(e))return"string"===t?[]:[e.length];if(Ks(e)){const t=e.channels||"RGBA";return[e.height,e.width*t.length]}if(qs(e))return[e.buffer.size/(null==t?4:I(t))];if(!Array.isArray(e))return[];const r=[];for(;Array.isArray(n)||ps(n)&&"string"!==t;)r.push(n.length),n=n[0];return Array.isArray(e)&&J().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&ua(e,r,[]),r}function ua(e,t,n){if(n=n||[],!Array.isArray(e)&&!ps(e))return void u(0===t.length,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);u(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),u(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const r=t.slice(1);for(let s=0;s<e.length;++s)ua(e[s],r,n.concat(s))}function ca(e,t,n,r){if("string_or_numeric"!==e){if(null==e)throw new Error("Expected dtype cannot be null.");if("numeric"!==e&&e!==t||"numeric"===e&&"string"===t)throw new Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function ha(e,t,n,r="numeric"){if(e instanceof Ms())return ca(r,e.dtype,t,n),e;let s=R(e);if("string"!==s&&["bool","int32","float32"].indexOf(r)>=0&&(s=r),ca(r,s,t,n),null==e||!ps(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e){const r=null==e?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}const a=la(e,s);ps(e)||Array.isArray(e)||(e=[e]);const o="string"!==s?cs(e,s):gs(e,[],!0);return ta.makeTensor(o,a,s)}function da(e,t,n,r="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,s)=>ha(e,`${t}[${s}]`,n,r))}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ia.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")}),ia.registerFlag("IS_BROWSER",()=>aa()),ia.registerFlag("IS_NODE",()=>"undefined"!=typeof process&&void 0!==process.versions&&void 0!==process.versions.node),ia.registerFlag("IS_CHROME",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),ia.registerFlag("IS_SAFARI",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),ia.registerFlag("PROD",()=>!1),ia.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>ia.getBool("DEBUG")),ia.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0),ia.registerFlag("IS_TEST",()=>!1),ia.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>ia.getBool("DEBUG")),ia.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1),ia.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1),ia.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);const fa="__op";function pa(e){const t=Object.keys(e);if(1!==t.length)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const r=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n+=fa;const s=(...e)=>{ta.startScope(n);try{const t=r(...e);return q(t)&&console.error("Cannot return a Promise inside of tidy."),ta.endScope(t),t}catch(t){throw ta.endScope(null),t}};return Object.defineProperty(s,"name",{value:n,configurable:!0}),s}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ga=pa({complex_:function(e,t){const n=ha(e,"real","complex"),r=ha(t,"imag","complex");c(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);const s={real:n,imag:r};return ta.runKernel(Ae,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ma(e,t,n,r){if(null==r)r=R(e);else if("complex64"===r)throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(qs(e)||Ks(e)){if("float32"!==r&&"int32"!==r)throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return ta.backend.createTensorFromGPUData(e,t||n,r)}if(!ps(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e)throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(null!=t){W(t);const e=d(t),r=d(n);u(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let s=0;s<n.length;++s){const e=n[s],r=s!==n.length-1||e!==d(t.slice(s));u(n[s]===t[s]||!r,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return ps(e)||Array.isArray(e)||(e=[e]),t=t||n,e="string"!==r?cs(e,r):gs(e,[],!0),ta.makeTensor(e,t,r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ba(e,t,n){return ma(e,t,la(e,n),n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ya={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class wa{static join(e){return new wa(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,null==e)return;if(e instanceof Array||(e=[e]),0===(e=e.map(e=>ps(e)?e.buffer:e)).length)return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){const r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const s=t+r.byteLength;this.shards.push({buffer:r,start:t,end:s}),t=s}0===this.shards.length&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(0===this.shards.length)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),(t=Math.min(this.byteLength,t))<=e)return new ArrayBuffer(0);const n=this.findShardForByte(e);if(-1===n)throw new Error(`Could not find start shard for byte ${e}`);const r=new ArrayBuffer(t-e),s=new Uint8Array(r);let a=0;for(let o=n;o<this.shards.length;o++){const n=this.shards[o],r=e+a-n.start,i=a,l=Math.min(t,n.end)-n.start,u=new Uint8Array(n.buffer,r,l-r);if(s.set(u,i),a+=u.length,t<n.end)break}return r}findShardForByte(e){if(0===this.shards.length||e<0||e>=this.byteLength)return-1;if(null!=this.bufferUniformSize)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:e>=t.end?1:0}if(0===t(this.shards[this.previousShardIndex]))return this.previousShardIndex;const n=function(e,t){let n=0,r=e.length;for(;n<=r;){const s=Math.floor((r-n)/2)+n,a=t(e[s]);if(0===a)return s;a<0?r=s:n=s+1}return-1}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(this.shards,t);return-1===n?-1:(this.previousShardIndex=n,this.previousShardIndex)}}function va(){return ta}function ka(e,t){return ta.tidy(e,t)}function xa(e){Hs(e).forEach(e=>e.dispose())}function Ea(e){return ta.keep(e)}function Sa(){return ta.backendName}function $a(e,t,n=1){return ta.registerBackend(e,t,n)}function Aa(){return ta.backend}function _a(e,t){const n=new wa(e),r={};let s=0;for(const a of t){const e=Ia(a,(e,t)=>n.slice(s+e,s+t));r[a.name]=Ta(a,n.slice(s,s+e)),s+=e}return r}function Ia(e,t){const n=d(e.shape);let r;if("quantization"in e){const t=e.quantization;r=ya[t.dtype]}else{if("string"===e.dtype){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(t(e,e+4))[0];return e}r=ya[e.dtype]}return n*r}async function Ma(e,t){const n=d(e.shape);let r;if("quantization"in e){const t=e.quantization;r=ya[t.dtype]}else{if("string"===e.dtype){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(await t(e,e+4))[0];return e}r=ya[e.dtype]}return n*r}function Ta(e,t){const n=e.name,r=e.dtype,s=e.shape,a=d(s);let o,i=0;if("quantization"in e){const s=e.quantization;if("uint8"===s.dtype||"uint16"===s.dtype){if(!("min"in s)||!("scale"in s))throw new Error(`Weight ${e.name} with quantization ${s.dtype} doesn't have corresponding metadata min and scale.`)}else{if("float16"!==s.dtype)throw new Error(`Weight ${e.name} has unknown quantization dtype ${s.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);if("float32"!==r)throw new Error(`Weight ${e.name} is quantized with ${s.dtype} which only supports weights of type float32 not ${r}.`)}const l=ya[s.dtype],u="uint8"===s.dtype?new Uint8Array(t):new Uint16Array(t);if("float32"===r)if("uint8"===s.dtype||"uint16"===s.dtype){o=new Float32Array(u.length);for(let e=0;e<u.length;e++){const t=u[e];o[e]=t*s.scale+s.min}}else{if("float16"!==s.dtype)throw new Error(`Unsupported quantization type ${s.dtype} for weight type float32.`);{const e=function(){const e=function(){const e=e=>{let t=e<<13,n=0;for(;!(8388608&t);)n-=8388608,t<<=1;return t&=-8388609,n+=947912704,t|n},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}(),t=function(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}(),n=function(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}();return r=>{const s=new ArrayBuffer(4*r.length),a=new Uint32Array(s);for(let o=0;o<r.length;o++){const s=r[o],i=e[n[s>>10]+(1023&s)]+t[s>>10];a[o]=i}return new Float32Array(s)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */();o=e(u)}}else{if("int32"!==r)throw new Error(`Unsupported dtype in weight '${n}': ${r}`);if("uint8"!==s.dtype&&"uint16"!==s.dtype)throw new Error(`Unsupported quantization type ${s.dtype} for weight type int32.`);o=new Int32Array(u.length);for(let e=0;e<u.length;e++){const t=u[e];o[e]=Math.round(t*s.scale+s.min)}}i+=a*l}else if("string"===r){const n=d(e.shape);o=[];for(let e=0;e<n;e++){const e=new Uint32Array(t.slice(i,i+4))[0];i+=4;const n=new Uint8Array(t.slice(i,i+e));o.push(n),i+=e}}else{const e=ya[r];if("float32"===r)o=new Float32Array(t);else if("int32"===r)o=new Int32Array(t);else{if("bool"!==r){if("complex64"===r){o=new Float32Array(t);const e=new Float32Array(o.length/2),n=new Float32Array(o.length/2);for(let t=0;t<e.length;t++)e[t]=o[2*t],n[t]=o[2*t+1];const r=ba(e,s,"float32"),a=ba(n,s,"float32"),i=ga(r,a);return r.dispose(),a.dispose(),i}throw new Error(`Unsupported dtype in weight '${n}': ${r}`)}o=new Uint8Array(t)}i+=a*e}return ba(o,s,r)}async function Na(e,t,n){let r=new Uint8Array(t);for(;r.byteLength<n;){const{done:t,value:s}=await e.read();if(t&&null==s){const e=n-r.byteLength;throw new Error(`Reader is done but ${e} bytes are still expected`)}const a=new Uint8Array(r.length+s.byteLength);a.set(r,0),a.set(new Uint8Array(s),r.length),r=a}return r.buffer}async function Da(e,t){const n={},r=e.getReader();let s=new ArrayBuffer(0);for(const a of t){const e=await Ma(a,async(e,t)=>(s=await Na(r,s,t),s.slice(e,t)));s=await Na(r,s,e);const t=s.slice(0,e);s=s.slice(e);const o=Ta(a,t);if(n[a.name]=o,"webgpu"===Sa()){const e=Aa();"uploadToGPU"in e&&d(o.shape)>=J().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&e.uploadToGPU(o.dataId)}}return n}function Ra(e){if(null===e)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});const r=new Uint8Array(t);let s=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),s),s+=e.byteLength}),r.buffer}const Ba="undefined"!=typeof Buffer&&("undefined"==typeof Blob||"undefined"==typeof atob||"undefined"==typeof btoa);function Fa(e){return Ba?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function Ca(e){for(e=e.trim();e.endsWith("/");)e=e.slice(0,e.length-1);const t=e.split("/");return t[t.length-1]}function Oa(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return null!=e.signature&&(n.signature=e.signature),null!=e.userDefinedMetadata&&(n.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(n.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(n.initializerSignature=e.initializerSignature),null!=e.trainingConfig&&(n.trainingConfig=e.trainingConfig),n}function Pa(e,t,n){const r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(null!=e.trainingConfig&&(r.trainingConfig=e.trainingConfig),null!=e.weightsManifest){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return null!=e.signature&&(r.signature=e.signature),null!=e.userDefinedMetadata&&(r.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(r.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(r.initializerSignature=e.initializerSignature),r}async function za(e,t){let n,r;return null!=e.weightsManifest&&([n,r]=await t(e.weightsManifest)),Pa(e,n,r)}function La(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:null==e.modelTopology?0:Fa(JSON.stringify(e.modelTopology)),weightSpecsBytes:null==e.weightSpecs?0:Fa(JSON.stringify(e.weightSpecs)),weightDataBytes:null==e.weightData?0:new wa(e.weightData).byteLength}}function Ua(e){const t=[];for(const n of e)t.push(...n.weights);return t}class Wa{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return null==Wa.instance&&(Wa.instance=new Wa),Wa.instance}static registerSaveRouter(e){Wa.getInstance().saveRouters.push(e)}static registerLoadRouter(e){Wa.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return Wa.getHandlers(e,"save")}static getLoadHandlers(e,t){return Wa.getHandlers(e,"load",t)}static getHandlers(e,t,n){const r=[];return("load"===t?Wa.getInstance().loadRouters:Wa.getInstance().saveRouters).forEach(t=>{const s=t(e,n);null!==s&&r.push(s)}),r}}const Ga="tensorflowjs",Ka="models_store",qa="model_info_store";function ja(){if(!J().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e="undefined"==typeof window?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(null==t)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Va(e){const t=e.result;t.createObjectStore(Ka,{keyPath:"modelPath"}),t.createObjectStore(qa,{keyPath:"modelPath"})}class Ha{constructor(e){if(this.indexedDB=ja(),null==e||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{const r=this.indexedDB.open(Ga,1);r.onupgradeneeded=()=>Va(r),r.onsuccess=()=>{const s=r.result;if(null==t){const t=s.transaction(Ka,"readonly"),r=t.objectStore(Ka).get(this.modelPath);r.onsuccess=()=>{if(null==r.result)return s.close(),n(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(s.close(),n(r.error)),t.oncomplete=()=>s.close()}else{t.weightData=wa.join(t.weightData);const r=La(t),o=s.transaction(qa,"readwrite");let i,l,u=o.objectStore(qa);try{i=u.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(a){return n(a)}i.onsuccess=()=>{l=s.transaction(Ka,"readwrite");const i=l.objectStore(Ka);let c;try{c=i.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(a){return n(a)}c.onsuccess=()=>e({modelArtifactsInfo:r}),c.onerror=e=>{u=o.objectStore(qa);const t=u.delete(this.modelPath);t.onsuccess=()=>(s.close(),n(c.error)),t.onerror=e=>(s.close(),n(c.error))}},i.onerror=e=>(s.close(),n(i.error)),o.oncomplete=()=>{null==l?s.close():l.oncomplete=()=>s.close()}}},r.onerror=e=>n(r.error)})}}Ha.URL_SCHEME="indexeddb://";const Ja=e=>{return J().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Ha.URL_SCHEME)?(t=e.slice(Ha.URL_SCHEME.length),new Ha(t)):null;var t};Wa.registerSaveRouter(Ja),Wa.registerLoadRouter(Ja);class Za{constructor(){this.indexedDB=ja()}async listModels(){return new Promise((e,t)=>{const n=this.indexedDB.open(Ga,1);n.onupgradeneeded=()=>Va(n),n.onsuccess=()=>{const r=n.result,s=r.transaction(qa,"readonly"),a=s.objectStore(qa).getAll();a.onsuccess=()=>{const t={};for(const e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),s.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){var t;return e=(t=e).startsWith(Ha.URL_SCHEME)?t.slice(Ha.URL_SCHEME.length):t,new Promise((t,n)=>{const r=this.indexedDB.open(Ga,1);r.onupgradeneeded=()=>Va(r),r.onsuccess=()=>{const s=r.result,a=s.transaction(qa,"readwrite"),o=a.objectStore(qa),i=o.get(e);let l;i.onsuccess=()=>{if(null==i.result)return s.close(),n(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const r=o.delete(e),a=()=>{l=s.transaction(Ka,"readwrite");const r=l.objectStore(Ka).delete(e);r.onsuccess=()=>t(i.result.modelArtifactsInfo),r.onerror=e=>n(i.error)};r.onsuccess=a,r.onerror=e=>(a(),s.close(),n(i.error))}},i.onerror=e=>(s.close(),n(i.error)),a.oncomplete=()=>{null==l?s.close():l.oncomplete=()=>s.close()}},r.onerror=e=>n(r.error)})}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ya="/",Xa="tensorflowjs_models",Qa="info",eo="model_topology",to="weight_specs",no="weight_data",ro="model_metadata";function so(e){return{info:[Xa,e,Qa].join(Ya),topology:[Xa,e,eo].join(Ya),weightSpecs:[Xa,e,to].join(Ya),weightData:[Xa,e,no].join(Ya),modelMetadata:[Xa,e,ro].join(Ya)}}function ao(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function oo(e){const t=e.split(Ya);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Ya)}class io{constructor(e){if(!J().getBool("IS_BROWSER")||"undefined"==typeof window||void 0===window.localStorage)throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,null==e||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=so(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(e.modelTopology),r=JSON.stringify(e.weightSpecs),s=La(e),a=wa.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(s)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,r),this.LS.setItem(this.keys.weightData,function(e){if(Ba)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let r=0,s=t.length;r<s;r++)n+=String.fromCharCode(t[r]);return btoa(n)}(a));const t={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:null!=e.signature?e.signature:void 0,userDefinedMetadata:null!=e.userDefinedMetadata?e.userDefinedMetadata:void 0,modelInitializer:null!=e.modelInitializer?e.modelInitializer:void 0,initializerSignature:null!=e.initializerSignature?e.initializerSignature:void 0,trainingConfig:null!=e.trainingConfig?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(t)),{modelArtifactsInfo:s}}catch(t){throw ao(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${s.modelTopologyBytes}, weightSpecsBytes=${s.weightSpecsBytes}, weightDataBytes=${s.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(null==e)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if("JSON"!==e.modelTopologyType)throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(null==n)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(null==r)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;const s=this.LS.getItem(this.keys.modelMetadata);if(null!=s){const e=JSON.parse(s);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,null!=e.signature&&(t.signature=e.signature),null!=e.userDefinedMetadata&&(t.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(t.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(t.initializerSignature=e.initializerSignature),null!=e.trainingConfig&&(t.trainingConfig=e.trainingConfig)}const a=this.LS.getItem(this.keys.weightData);if(null==a)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=function(e){if(Ba){const t=Buffer.from(e,"base64");return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let r=0;r<t.length;++r)n.set([t.charCodeAt(r)],r);return n.buffer}(a),t}}io.URL_SCHEME="localstorage://";const lo=e=>{return J().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(io.URL_SCHEME)?(t=e.slice(io.URL_SCHEME.length),new io(t)):null;var t};Wa.registerSaveRouter(lo),Wa.registerLoadRouter(lo);class uo{constructor(){u(J().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),u("undefined"==typeof window||void 0!==window.localStorage,()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=Xa+Ya,n=Ya+Qa;for(let r=0;r<this.LS.length;++r){const s=this.LS.key(r);if(s.startsWith(t)&&s.endsWith(n)){e[oo(s)]=JSON.parse(this.LS.getItem(s))}}return e}async removeModel(e){var t;const n=so(e=(t=e).startsWith(io.URL_SCHEME)?t.slice(io.URL_SCHEME.length):t);if(null==this.LS.getItem(n.info))throw new Error(`Cannot find model at path '${e}'`);const r=JSON.parse(this.LS.getItem(n.info));return ao(n),r}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const co="://";class ho{constructor(){this.managers={}}static getInstance(){return null==ho.instance&&(ho.instance=new ho),ho.instance}static registerManager(e,t){u(null!=e,()=>"scheme must not be undefined or null."),e.endsWith(co)&&(e=e.slice(0,e.indexOf(co))),u(e.length>0,()=>"scheme must not be an empty string.");const n=ho.getInstance();u(null==n.managers[e],()=>`A model store manager is already registered for scheme '${e}'.`),n.managers[e]=t}static getManager(e){const t=ho.getInstance().managers[e];if(null==t)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(ho.getInstance().managers)}}function fo(e){if(-1===e.indexOf(co))throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${ho.getSchemes().join(",")}`);return{scheme:e.split(co)[0],path:e.split(co)[1]}}async function po(e,t,n=!1){u(e!==t,()=>`Old path and new path are the same: '${e}'`);const r=Wa.getLoadHandlers(e);u(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),u(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);const s=r[0],a=Wa.getSaveHandlers(t);u(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),u(a.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);const o=a[0],i=fo(e).scheme,l=fo(e).path,c=i===fo(e).scheme,h=await s.load();n&&c&&await ho.getManager(i).removeModel(l);const d=await o.save(h);return n&&!c&&await ho.getManager(i).removeModel(l),d.modelArtifactsInfo}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class go{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return null==this.textEncoder&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){"undefined"!=typeof window&&J().getBool("USE_SETTIMEOUTCUSTOM")?(this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();(0,this.functionRefs[e.data.index])(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))):setTimeout(e,t)}isTypedArray(e){return vr(e)}}if(J().get("IS_BROWSER")){J().setPlatform("browser",new go);try{ho.registerManager(io.URL_SCHEME,new uo)}catch(Up){}try{ho.registerManager(Ha.URL_SCHEME,new Za)}catch(Up){}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mo=()=>require("node-fetch");let bo;class yo{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return null!=J().global.fetch?J().global.fetch(e,t):(null==bo&&(bo=mo()),bo(e,t))}now(){const e=process.hrtime();return 1e3*e[0]+e[1]/1e6}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return 0===e.length?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function wo(e,t="float32",n){return t=t||"float32",W(e),new $s(e,t,n)}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */J().get("IS_NODE")&&!J().get("IS_BROWSER")&&J().setPlatform("node",new yo);const vo=pa({cast_:function(e,t){const n=ha(e,"x","cast");if(!A(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if("string"===t&&"string"!==n.dtype||"string"!==t&&"string"===n.dtype)throw new Error("Only strings can be casted to strings");const r={x:n},s={dtype:t};return ta.runKernel(Ee,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ko=pa({clone_:function(e){const t={x:ha(e,"x","clone","string_or_numeric")};return ta.runKernel(gt,t)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xo(e,t=!1){console.log(e.toString(t))}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ea();_s={buffer:wo,cast:vo,clone:ko,print:xo};const Eo=pa({add_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t){let n=ha(e,"a","add"),r=ha(t,"b","add");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(re,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const So=pa({floorDiv_:function(e,t){let n=ha(e,"a","floorDiv"),r=ha(t,"b","floorDiv");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(ut,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $o=pa({div_:function(e,t){let n=ha(e,"a","div"),r=ha(t,"b","div");if([n,r]=js(n,r),"int32"===n.dtype&&"int32"===r.dtype)return So(n,r);const s={a:n,b:r};return ta.runKernel(Ze,s,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ao=pa({mul_:function(e,t){let n=ha(e,"a","mul"),r=ha(t,"b","mul");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(qt,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _o=pa({abs_:function(e){const t=ha(e,"x","abs");if("complex64"===t.dtype){const e={x:t};return ta.runKernel(_e,e)}{const e={x:t};return ta.runKernel(ee,e)}}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Io=pa({acos_:function(e){const t={x:ha(e,"x","acos")};return ta.runKernel(te,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mo=pa({acosh_:function(e){const t={x:ha(e,"x","acosh")};return ta.runKernel(ne,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const To=pa({addN_:function(e){u(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),u(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((e,t)=>ha(e,`tensors${t}`,"addN")),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(e=>{if(!g(e.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const r=t;return ta.runKernel(se,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const No=pa({all_:function(e,t=null,n=!1){const r={x:ha(e,"x","all","bool")},s={axis:t,keepDims:n};return ta.runKernel(ae,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Do=pa({any_:function(e,t=null,n=!1){const r={x:ha(e,"x","any","bool")},s={axis:t,keepDims:n};return ta.runKernel(oe,r,s)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ro=pa({argMax_:function(e,t=0){const n={x:ha(e,"x","argMax")},r={axis:t};return ta.runKernel(ie,n,r)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bo=pa({argMin_:function(e,t=0){const n={x:ha(e,"x","argMin")},r={axis:t};return ta.runKernel(le,n,r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fo=pa({asin_:function(e){const t={x:ha(e,"x","asin")};return ta.runKernel(ue,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Co=pa({asinh_:function(e){const t={x:ha(e,"x","asinh")};return ta.runKernel(ce,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oo=pa({atan_:function(e){const t={x:ha(e,"x","atan")};return ta.runKernel(he,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Po=pa({atan2_:function(e,t){let n=ha(e,"a","atan2"),r=ha(t,"b","atan2");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(fe,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zo=pa({atanh_:function(e){const t={x:ha(e,"x","atanh")};return ta.runKernel(de,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lo(e,t,n,r,s="NHWC",a){return Go(e,[...t,e[3]],n,a,r,null,null,Qo(s))}function Uo(e,t,n,r,s,a,o="channelsLast"){const[i,l]=jo(t);let u;if("channelsLast"===o)u=[i,l,e[3],e[3]];else{if("channelsFirst"!==o)throw new Error(`Unknown dataFormat ${o}`);u=[i,l,e[1],e[1]]}return Go(e,u,n,r,s,a,!1,o)}function Wo(e,t,n,r,s,a,o="NDHWC"){const[i,l,u]=Vo(t);let c,h;if("NDHWC"===o)h="channelsLast",c=[i,l,u,e[4],e[4]];else{if("NCDHW"!==o)throw new Error(`Unknown dataFormat ${o}`);h="channelsFirst",c=[i,l,u,e[1],e[1]]}return Ko(e,c,n,r,s,!1,h,a)}function Go(e,t,n,r,s,a,o=!1,i="channelsLast"){let[l,u,c,h]=[-1,-1,-1,-1];if("channelsLast"===i)[l,u,c,h]=e;else{if("channelsFirst"!==i)throw new Error(`Unknown dataFormat ${i}`);[l,h,u,c]=e}const[d,f,,p]=t,[g,m]=jo(n),[b,y]=jo(r),w=Ho(d,b),v=Ho(f,y),{padInfo:k,outHeight:x,outWidth:E}=function(e,t,n,r,s,a,o,i,l){let u,c,h;if("number"==typeof e){u={top:e,bottom:e,left:e,right:e,type:0===e?"VALID":"NUMBER"};const s=function(e,t,n,r,s){null==r&&(r=qo(e,t,n));const a=e[0],o=e[1],i=Jo((a-t+2*r)/n+1,s),l=Jo((o-t+2*r)/n+1,s);return[i,l]}([t,n],a,r,e,i);c=s[0],h=s[1]}else if("same"===e){c=Math.ceil(t/r),h=Math.ceil(n/s);const e=Math.max(0,(c-1)*r+a-t),i=Math.max(0,(h-1)*s+o-n),l=Math.floor(e/2),d=e-l,f=Math.floor(i/2);u={top:l,bottom:d,left:f,right:i-f,type:"SAME"}}else if("valid"===e)u={top:0,bottom:0,left:0,right:0,type:"VALID"},c=Math.ceil((t-a+1)/r),h=Math.ceil((n-o+1)/s);else{if("object"!=typeof e)throw Error(`Unknown padding parameter: ${e}`);{const d="channelsLast"===l?e[1][0]:e[2][0],f="channelsLast"===l?e[1][1]:e[2][1],p="channelsLast"===l?e[2][0]:e[3][0],g="channelsLast"===l?e[2][1]:e[3][1];u={top:d,bottom:f,left:p,right:g,type:0===d&&0===f&&0===p&&0===g?"VALID":"EXPLICIT"},c=Jo((t-a+d+f)/r+1,i),h=Jo((n-o+p+g)/s+1,i)}}return{padInfo:u,outHeight:c,outWidth:h}}(s,u,c,g,m,w,v,a,i),S=o?p*h:p;let $;return"channelsFirst"===i?$=[l,S,x,E]:"channelsLast"===i&&($=[l,x,E,S]),{batchSize:l,dataFormat:i,inHeight:u,inWidth:c,inChannels:h,outHeight:x,outWidth:E,outChannels:S,padInfo:k,strideHeight:g,strideWidth:m,filterHeight:d,filterWidth:f,effectiveFilterHeight:w,effectiveFilterWidth:v,dilationHeight:b,dilationWidth:y,inShape:e,outShape:$,filterShape:t}}function Ko(e,t,n,r,s,a=!1,o="channelsLast",i){let[l,u,c,h,d]=[-1,-1,-1,-1,-1];if("channelsLast"===o)[l,u,c,h,d]=e;else{if("channelsFirst"!==o)throw new Error(`Unknown dataFormat ${o}`);[l,d,u,c,h]=e}const[f,p,g,,m]=t,[b,y,w]=Vo(n),[v,k,x]=Vo(r),E=Ho(f,v),S=Ho(p,k),$=Ho(g,x),{padInfo:A,outDepth:_,outHeight:I,outWidth:M}=function(e,t,n,r,s,a,o,i,l,u,c){let h,d,f,p;"valid"===e&&(e=0);if("number"==typeof e){h={top:e,bottom:e,left:e,right:e,front:e,back:e,type:0===e?"VALID":"NUMBER"};const g=function(e,t,n,r,s,a){null==s&&(s=qo(e,t[0],r[0]));const o=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*s>=t[i]&&(o[i]=Jo((e[i]-t[i]+2*s)/r[i]+1,a));return o}([t,n,r,1],[i,l,u],1,[s,a,o],e,c);d=g[0],f=g[1],p=g[2]}else{if("same"!==e)throw Error(`Unknown padding parameter: ${e}`);{d=Math.ceil(t/s),f=Math.ceil(n/a),p=Math.ceil(r/o);const e=(d-1)*s+i-t,c=(f-1)*a+l-n,g=(p-1)*o+u-r,m=Math.floor(e/2),b=e-m,y=Math.floor(c/2),w=c-y,v=Math.floor(g/2);h={top:y,bottom:w,left:v,right:g-v,front:m,back:b,type:"SAME"}}}return{padInfo:h,outDepth:d,outHeight:f,outWidth:p}}(s,u,c,h,b,y,w,E,S,$,i),T=a?m*d:m;let N;return"channelsFirst"===o?N=[l,T,_,I,M]:"channelsLast"===o&&(N=[l,_,I,M,T]),{batchSize:l,dataFormat:o,inDepth:u,inHeight:c,inWidth:h,inChannels:d,outDepth:_,outHeight:I,outWidth:M,outChannels:T,padInfo:A,strideDepth:b,strideHeight:y,strideWidth:w,filterDepth:f,filterHeight:p,filterWidth:g,effectiveFilterDepth:E,effectiveFilterHeight:S,effectiveFilterWidth:$,dilationDepth:v,dilationHeight:k,dilationWidth:x,inShape:e,outShape:N,filterShape:t}}function qo(e,t,n,r=1){const s=Ho(t,r);return Math.floor((e[0]*(n-1)-n+s)/2)}function jo(e){return"number"==typeof e?[e,e,e]:2===e.length?[e[0],e[1],1]:e}function Vo(e){return"number"==typeof e?[e,e,e]:e}function Ho(e,t){return t<=1?e:e+(e-1)*(t-1)}function Jo(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function Zo(e){const[t,n,r]=jo(e);return 1===t&&1===n&&1===r}function Yo(e,t){return Zo(e)||Zo(t)}function Xo(e){return jo(e).every(e=>e>0)}function Qo(e){if("NHWC"===e)return"channelsLast";if("NCHW"===e)return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function ei(e,t,n){if(null!=n){if("string"==typeof t)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if("number"==typeof t)u(m(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else{if("object"!=typeof t)throw Error(`Error in ${e}: Unknown padding parameter: ${t}`);t.forEach(t=>{t.forEach(t=>{u(m(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})})}}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ti=pa({reshape_:function(e,t){const n={x:ha(e,"x","reshape","string_or_numeric")},r={shape:t};return ta.runKernel(dn,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ni=pa({avgPool_:function(e,t,n,r,s){const a=ha(e,"x","avgPool","float32");u(Yo(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=a,i=!1;3===a.rank&&(i=!0,o=ti(a,[1,a.shape[0],a.shape[1],a.shape[2]])),u(4===o.rank,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),ei("avgPool",r,s);const l={x:o},c={filterSize:t,strides:n,pad:r,dimRoundingMode:s};let h=ta.runKernel(pe,l,c);return h=vo(h,a.dtype),i?ti(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ri=pa({avgPool3d_:function(e,t,n,r,s,a="NDHWC"){const o=ha(e,"x","avgPool3d","float32");let i=o,l=!1;4===o.rank&&(l=!0,i=ti(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),u(5===i.rank,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),u("NDHWC"===a,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),u("number"==typeof n&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),ei("avgPool3d",r,s);const c={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:a};let d=ta.runKernel(me,c,h);return d=vo(d,i.dtype),l?ti(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const si=pa({concat_:function(e,t=0){u(e.length>=1,()=>"Pass at least one tensor to concat");const n=da(e,"tensors","concat","string_or_numeric");if("complex64"===n[0].dtype&&n.forEach(e=>{if("complex64"!==e.dtype)throw new Error(`Cannot concatenate complex64 tensors with a tensor\n          with dtype ${e.dtype}. `)}),1===n.length)return ko(n[0]);const r=n,s={axis:t};return ta.runKernel(Ie,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ai=pa({matMul_:function(e,t,n=!1,r=!1){let s=ha(e,"a","matMul"),a=ha(t,"b","matMul");[s,a]=js(s,a);const o={a:s,b:a},i={transposeA:n,transposeB:r};return ta.runKernel(ye,o,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oi=pa({sigmoid_:function(e){const t={x:ha(e,"x","sigmoid","float32")};return ta.runKernel(Tn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ii=pa({slice_:function(e,t,n){const r=ha(e,"x","slice","string_or_numeric");if(0===r.rank)throw new Error("Slicing scalar is not possible");const s={x:r},a={begin:t,size:n};return ta.runKernel(An,s,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const li=pa({tanh_:function(e){const t={x:ha(e,"x","tanh","float32")};return ta.runKernel(Yn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ui=pa({basicLSTMCell_:function(e,t,n,r,s,a){const o=ha(e,"forgetBias","basicLSTMCell"),i=ha(t,"lstmKernel","basicLSTMCell"),l=ha(n,"lstmBias","basicLSTMCell"),u=ha(r,"data","basicLSTMCell"),c=ha(s,"c","basicLSTMCell"),h=ha(a,"h","basicLSTMCell"),d=si([u,h],1),f=ai(d,i),p=Eo(f,l),g=p.shape[0],m=p.shape[1]/4,b=[g,m],y=ii(p,[0,0],b),w=ii(p,[0,m],b),v=ii(p,[0,2*m],b),k=ii(p,[0,3*m],b),x=Eo(Ao(oi(y),li(w)),Ao(c,oi(Eo(o,v))));return[x,Ao(li(x),oi(k))]}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ci=pa({batchToSpaceND_:function(e,t,n){const r=ha(e,"x","batchToSpaceND"),s=t.reduce((e,t)=>e*t);u(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),u(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),u(r.shape[0]%s===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${s}`);const a={x:r},o={blockShape:t,crops:n};return ta.runKernel(we,a,o)}});const hi=pa({batchNorm_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r,s,a){null==a&&(a=.001);const o=ha(e,"x","batchNorm"),i=ha(t,"mean","batchNorm"),l=ha(n,"variance","batchNorm");let c,h;null!=s&&(c=ha(s,"scale","batchNorm")),null!=r&&(h=ha(r,"offset","batchNorm")),u(i.rank===l.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),u(null==h||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),u(null==c||i.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d={x:function(e){let t;return t=0===e.rank||1===e.rank?ti(e,[1,1,1,e.size]):2===e.rank?ti(e,[1,1,e.shape[0],e.shape[1]]):3===e.rank?ti(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}(o),scale:c,offset:h,mean:i,variance:l},f={varianceEpsilon:a},p=ta.runKernel(ct,d,f);return ti(p,o.shape)}});const di=pa({batchNorm2d_:function(e,t,n,r,s,a){const o=ha(e,"x","batchNorm"),i=ha(t,"mean","batchNorm"),l=ha(n,"variance","batchNorm");let c,h;return null!=s&&(c=ha(s,"scale","batchNorm")),null!=r&&(h=ha(r,"offset","batchNorm")),u(2===o.rank,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),u(2===i.rank||1===i.rank,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),u(2===l.rank||1===l.rank,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${l.rank}.`),null!=c&&u(2===c.rank||1===c.rank,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${c.rank}.`),null!=h&&u(2===h.rank||1===h.rank,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),hi(o,i,l,h,c,a)}});const fi=pa({batchNorm3d_:function(e,t,n,r,s,a){const o=ha(e,"x","batchNorm"),i=ha(t,"mean","batchNorm"),l=ha(n,"variance","batchNorm");let c,h;return null!=s&&(c=ha(s,"scale","batchNorm")),null!=r&&(h=ha(r,"offset","batchNorm")),u(3===o.rank,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),u(3===i.rank||1===i.rank,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),u(3===l.rank||1===l.rank,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${l.rank}.`),null!=c&&u(3===c.rank||1===c.rank,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${c.rank}.`),null!=h&&u(3===h.rank||1===h.rank,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),hi(o,i,l,h,c,a)}});const pi=pa({batchNorm4d_:function(e,t,n,r,s,a){const o=ha(e,"x","batchNorm"),i=ha(t,"mean","batchNorm"),l=ha(n,"variance","batchNorm");let c,h;return null!=s&&(c=ha(s,"scale","batchNorm")),null!=r&&(h=ha(r,"offset","batchNorm")),u(4===o.rank,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),u(4===i.rank||1===i.rank,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),u(4===l.rank||1===l.rank,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${l.rank}.`),null!=c&&u(4===c.rank||1===c.rank,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${c.rank}.`),null!=h&&u(4===h.rank||1===h.rank,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),hi(o,i,l,h,c,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gi=pa({bincount_:function(e,t,n){const r=ha(e,"x","bincount"),s=ha(t,"weights","bincount");u("int32"===r.dtype,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),u(n>=0,()=>`size must be non-negative, but got ${n}.`),u(s.size===r.size||0===s.size,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${s.shape}.`);const a={x:r,weights:s},o={size:n};return ta.runKernel(ve,a,o)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mi=pa({bitwiseAnd_:function(e,t){const n=ha(e,"x","bitwiseAnd"),r=ha(t,"y","bitwiseAnd");if(!g(n.shape,r.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${r.shape}`);if("int32"!==n.dtype||"int32"!==r.dtype)throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${r.dtype}`);const s={a:n,b:r};return ta.runKernel(ke,s)}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bi=pa({broadcastArgs_:function(e,t){const n=ha(e,"s0","broadcastArgs","int32"),r=ha(t,"s1","broadcastArgs","int32");if(1!==n.rank)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(1!==r.rank)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);const s={s0:n,s1:r};return ta.runKernel(xe,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yi=pa({broadcastTo_:function(e,t){let n=ha(e,"broadcastTo","x");const r=n.shape;if(W(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=ti(n,e)}const s=n.shape,a=Array.from(t);for(let l=t.length-1;l>=0;l--)if(s[l]===t[l])a[l]=1;else if(1!==n.shape[l])throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(0===a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length)return ko(n);const o={x:n},i={reps:a};return ta.runKernel(Xn,o,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wi=pa({ceil_:function(e){const t={x:ha(e,"x","ceil","float32")};return ta.runKernel(Se,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vi(e,t,n){W(e);const r={shape:e,value:t,dtype:n=n||R(t)};return ta.runKernel(ot,{},r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ki=pa({clipByValue_:function(e,t,n){const r=ha(e,"x","clipByValue");if(u(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return vi(r.shape,t,r.dtype);const s={x:r},a={clipValueMin:t,clipValueMax:n};return ta.runKernel($e,s,a)}});const xi=pa({concat1d_:function(e){return si(e,0)}});const Ei=pa({concat2d_:function(e,t){return si(e,t)}});const Si=pa({concat3d_:function(e,t){return si(e,t)}});const $i=pa({concat4d_:function(e,t){return si(e,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ai=pa({conv2d_:function(e,t,n,r,s="NHWC",a=[1,1],o){const i=ha(e,"x","conv2d","float32"),l=ha(t,"filter","conv2d","float32");let c=i,h=!1;3===i.rank&&(h=!0,c=ti(i,[1,i.shape[0],i.shape[1],i.shape[2]])),u(4===c.rank,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),u(4===l.rank,()=>`Error in conv2d: filter must be rank 4, but got rank ${l.rank}.`),ei("conv2d",r,o);const d="NHWC"===s?c.shape[3]:c.shape[1];u(d===l.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${l.shape[2]}.`),u(Yo(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),u(Xo(a),()=>"Error in conv2D: Dilated rates should be larger than 0."),u(Xo(n),()=>"Error in conv2D: Strides should be larger than 0.");const f={x:c,filter:l},p={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o},g=ta.runKernel(Me,f,p);return h?ti(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});const _i=pa({conv1d_:function(e,t,n,r,s="NWC",a=1,o){const i=ha(e,"x","conv1d"),l=ha(t,"filter","conv1d");let c=i,h=!1;2===i.rank&&(h=!0,c=ti(i,[1,i.shape[0],i.shape[1]])),u(3===c.rank,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),u(3===l.rank,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),ei("conv1d",r,o),u(c.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${l.shape[1]}.`),u(Yo(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),u(Xo(a),()=>"Error in conv1D: Dilated rates should be larger than 0."),u(Xo(n),()=>"Error in conv1D: Stride should be larger than 0."),u("NWC"===s,()=>`Error in conv1d: got dataFormat of ${s} but only NWC is currently supported.`);const d=ti(l,[1,l.shape[0],l.shape[1],l.shape[2]]),f=ti(c,[c.shape[0],1,c.shape[1],c.shape[2]]),p=Ai(f,d,[1,n],r,"NHWC",[1,a],o);return ti(p,h?[p.shape[2],p.shape[3]]:[p.shape[0],p.shape[2],p.shape[3]])}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ii=pa({conv2DBackpropInput_:function(e,t,n,r,s,a="NHWC",o){u(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,l=t,c=!1;3===t.rank&&(c=!0,l=ti(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),u(4===i.length,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),u(4===l.rank,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${l.rank}`),u(4===n.rank,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h="NHWC"===a?i[3]:i[1],d="NHWC"===a?l.shape[3]:l.shape[1];u(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),u(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),ei("conv2dDerInput",s,o);const f={dy:l,filter:n},p={strides:r,pad:s,dataFormat:a,dimRoundingMode:o,inputShape:i},g=ta.runKernel(Ne,f,p);return c?ti(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});const Mi=pa({conv2dTranspose_:function(e,t,n,r,s,a){const o=ha(e,"x","conv2dTranspose"),i=ha(t,"filter","conv2dTranspose");return Ii(n,o,i,r,s,"NHWC",a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ti=pa({conv3d_:function(e,t,n,r,s="NDHWC",a=[1,1,1]){const o=ha(e,"x","conv3d"),i=ha(t,"filter","conv3d");let l=o,c=!1;4===o.rank&&(c=!0,l=ti(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),u(5===l.rank,()=>`Error in conv3d: input must be rank 5, but got rank ${l.rank}.`),u(5===i.rank,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),u(l.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${l.shape[4]}) must match input depth for filter ${i.shape[3]}.`),u(Yo(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),u("NDHWC"===s,()=>`Error in conv3d: got dataFormat of ${s} but only NDHWC is currently supported.`),u(Xo(a),()=>"Error in conv3D: Dilated rates should be larger than 0."),u(Xo(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:l,filter:i},d={strides:n,pad:r,dataFormat:s,dilations:a},f=ta.runKernel(De,h,d);return c?ti(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ni=pa({conv3DBackpropInput_:function(e,t,n,r,s){u(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,i=!1;4===t.rank&&(i=!0,o=ti(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);const l=a[4],c=o.shape[4];u(5===a.length,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),u(5===o.rank,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),u(5===n.rank,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),u(l===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${l}) must match input depth for filter ${n.shape[3]}.`),u(c===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[4]}.`);const h={dy:o,filter:n},d={pad:s,strides:r,inputShape:a},f=ta.runKernel(Be,h,d);return i?ti(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}});const Di=pa({conv3dTranspose_:function(e,t,n,r,s){const a=ha(e,"x","conv3dTranspose"),o=ha(t,"filter","conv3dTranspose");return Ni(n,a,o,r,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ri=pa({cos_:function(e){const t={x:ha(e,"x","cos","float32")};return ta.runKernel(Fe,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bi=pa({cosh_:function(e){const t={x:ha(e,"x","cosh","float32")};return ta.runKernel(Ce,t)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fi=pa({cumprod_:function(e,t=0,n=!1,r=!1){const s={x:ha(e,"x","cumprod")},a={axis:t,exclusive:n,reverse:r};return ta.runKernel(Oe,s,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ci=pa({cumsum_:function(e,t=0,n=!1,r=!1){const s={x:ha(e,"x","cumsum")},a={axis:t,exclusive:n,reverse:r};return ta.runKernel(Pe,s,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oi=pa({denseBincount_:function(e,t,n,r=!1){const s=ha(e,"x","denseBincount"),a=ha(t,"weights","denseBincount");u("int32"===s.dtype,()=>`Error in denseBincount: input dtype must be int32, but got ${s.dtype}`),u(s.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${s.rank}.`),u(n>=0,()=>`size must be non-negative, but got ${n}.`),u(a.size===s.size||0===a.size,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${s.shape}, weights shape: ${a.shape}.`);const o={x:s,weights:a},i={size:n,binaryOutput:r};return ta.runKernel(Le,o,i)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pi=pa({depthToSpace_:function(e,t,n="NHWC"){const r=ha(e,"x","depthToSpace","float32"),s="NHWC"===n?r.shape[1]:r.shape[2],a="NHWC"===n?r.shape[2]:r.shape[3],o="NHWC"===n?r.shape[3]:r.shape[1];u(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),u(s*t>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${s} and ${t}  for depthToSpace with input shape\n    ${r.shape}`),u(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${a} and ${t} for depthToSpace with input shape\n        ${r.shape}`),u(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);const i={x:r},l={blockSize:t,dataFormat:n};return ta.runKernel(Ue,i,l)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zi=pa({depthwiseConv2d_:function(e,t,n,r,s="NHWC",a=[1,1],o){const i=ha(e,"x","depthwiseConv2d","float32"),l=ha(t,"filter","depthwiseConv2d","float32");let c=i,h=!1;3===i.rank&&(h=!0,c=ti(i,[1,i.shape[0],i.shape[1],i.shape[2]])),u(4===c.rank,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),u(4===l.rank,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${l.rank}.`);const d="NHWC"===s?c.shape[3]:c.shape[1];u(d===l.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${d}) must match the inChannels dimension in filter ${l.shape[2]}.`),ei("depthwiseConv2d",r,o);const f={x:c,filter:l},p={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o},g=ta.runKernel(We,f,p);return h?ti(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Li=pa({diag_:function(e){const t={x:ha(e,"x","diag")};return ta.runKernel(qe,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ui=pa({dilation2d_:function(e,t,n,r,s=[1,1],a="NHWC"){const o=ha(e,"x","dilation2d"),i=ha(t,"filter","dilation2d");u(3===o.rank||4===o.rank,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),u(3===i.rank,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),u("NHWC"===a,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let l=o,c=!1;3===o.rank&&(l=ti(o,[1,o.shape[0],o.shape[1],o.shape[2]]),c=!0),u(l.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${l.shape[3]} vs ${i.shape[2]}`);const h={x:l,filter:i},d={strides:n,pad:r,dilations:s},f=ta.runKernel(je,h,d);return c?ti(f,[f.shape[1],f.shape[2],f.shape[3]]):f}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wi(e,t){const n=e.length,r=[];for(let s=0;s<n;s++){const a=n-1-s,o=e[a]||1;(t[t.length-1-s]||1)>1&&1===o&&r.unshift(a)}return r}function Gi(e,t){const n=[];for(let r=0;r<t.length;r++){const s=e[e.length-r-1],a=t.length-r-1,o=t[a];(null==s||1===s&&o>1)&&n.unshift(a)}return n}function Ki(e,t){const n=Math.max(e.length,t.length),r=new Array(n);for(let s=0;s<n;s++){let a=e[e.length-s-1];null==a&&(a=1);let o=t[t.length-s-1];if(null==o&&(o=1),1===a)r[n-s-1]=o;else if(1===o)r[n-s-1]=a;else{if(a!==o){throw Error(`Operands could not be broadcast together with shapes ${e} and ${t}.`)}r[n-s-1]=a}}return r}const qi=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:Ki,getBroadcastDims:Wi,getReductionAxes:Gi},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ji=pa({equal_:function(e,t){let n=ha(e,"a","equal","string_or_numeric"),r=ha(t,"b","equal","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(tt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vi=pa({where_:function(e,t,n){const r=ha(t,"a","where"),s=ha(n,"b","where"),a=ha(e,"condition","where","bool"),o=Ki(Ki(a.shape,r.shape),s.shape),i={condition:yi(a,o),t:yi(r,o),e:yi(s,o)};return ta.runKernel(Sn,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hi=pa({zerosLike_:function(e){const t={x:ha(e,"x","zerosLike")};return ta.runKernel(ar,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ji=pa({divNoNan_:function(e,t){let n=ha(e,"a","div"),r=ha(t,"b","div");[n,r]=js(n,r);const s=$o(n,r),a=Hi(s),o=ji(r,a);return Vi(o,a,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zi=pa({dot_:function(e,t){const n=ha(e,"t1","dot"),r=ha(t,"t2","dot");u(!(1!==n.rank&&2!==n.rank||1!==r.rank&&2!==r.rank),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);const s=1===n.rank?n.size:n.shape[1],a=1===r.rank?r.size:r.shape[0];if(u(s===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${s} and ${a}.`),1===n.rank&&1===r.rank){const e=ti(n,[1,-1]),t=ti(r,[-1,1]),s=ai(e,t);return ti(s,[])}if(1===n.rank&&2===r.rank){const e=ti(n,[1,-1]),t=ti(r,[r.shape[0],r.shape[1]]),s=ai(e,t);return ti(s,[s.size])}if(2===n.rank&&1===r.rank){const e=ti(r,[-1,1]),t=ai(n,e);return ti(t,[t.size])}{const e=ti(r,[r.shape[0],r.shape[1]]);return ai(n,e)}}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yi=pa({einsum_:function(e,...t){const n=t.map((e,t)=>ha(e,`tensors${t}`,"einsum")),r={equation:e};return ta.runKernel(Ye,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xi=pa({elu_:function(e){const t={x:ha(e,"x","elu","float32")};return ta.runKernel(Xe,t)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qi=pa({ensureShape_:function(e,t){const n=ha(e,"x","ensureShape","string_or_numeric");if(!p(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const el=pa({erf_:function(e){let t=ha(e,"x","erf");u("int32"===t.dtype||"float32"===t.dtype,()=>"Input dtype must be `int32` or `float32`."),"int32"===t.dtype&&(t=vo(t,"float32"));const n={x:t};return ta.runKernel(et,n)}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tl(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function nl(e,t,n){const r=e.length+t.length,s=[];let a=0,o=0;for(let i=0;i<r;i++)-1===n.indexOf(i)?s.push(e[a++]):s.push(t[o++]);return s}function rl(e,t){const n=[],r=e.length;for(let s=0;s<r;s++)-1===t.indexOf(s)&&n.push(e[s]);return[n,t.map(t=>e[t])]}function sl(e,t){return nl(e,t.map(e=>1),t)}function al(e,t,n){u(tl(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function ol(e,t){if(tl(e,t))return null;const n=[];for(let r=0;r<t;++r)-1===e.indexOf(r)&&n.push(r);return e.forEach(e=>n.push(e)),n}function il(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function ll(e,t){const n=[];for(let r=t-e;r<t;++r)n.push(r);return n}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ul=pa({max_:function(e,t=null,n=!1){const r={x:ha(e,"x","max")},s={reductionIndices:t,keepDims:n};return ta.runKernel(Dt,r,s)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cl=pa({min_:function(e,t=null,n=!1){const r={x:ha(e,"x","min")},s={axis:t,keepDims:n};return ta.runKernel(Lt,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hl=pa({pow_:function(e,t){let n=ha(e,"base","pow"),r=ha(t,"exp","pow");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(tn,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dl(e,t){if((ps(e)&&"string"!==t||Array.isArray(e))&&"complex64"!==t)throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if("string"===t&&ps(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return ma(e,[],[],t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fl=pa({sqrt_:function(e){const t={x:ha(e,"x","sqrt","float32")};return ta.runKernel(Dn,t)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pl=pa({square_:function(e){const t=ha(e,"x","square");return ta.runKernel("Square",{x:t},{})}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gl=pa({sum_:function(e,t=null,n=!1){let r=ha(e,"x","sum");"bool"===r.dtype&&(r=vo(r,"int32"));const s={x:r},a={axis:t,keepDims:n};return ta.runKernel(Rn,s,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ml(e,t,n=null){if(0===e.rank)return _o(e);if(1!==e.rank&&null===n)return ml(ti(e,[-1]),t,n);if(1===e.rank||"number"==typeof n||Array.isArray(n)&&1===n.length){if(1===t)return gl(_o(e),n);if(t===1/0)return ul(_o(e),n);if(t===-1/0)return cl(_o(e),n);if("euclidean"===t||2===t)return fl(gl(hl(_o(e),dl(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&2===n.length){if(1===t)return ul(gl(_o(e),n[0]),n[1]-1);if(t===1/0)return ul(gl(_o(e),n[1]),n[0]);if(t===-1/0)return cl(gl(_o(e),n[1]),n[0]);if("fro"===t||"euclidean"===t)return fl(gl(pl(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const bl=pa({norm_:function(e,t="euclidean",n=null,r=!1){const s=ml(e=ha(e,"x","norm"),t,n);let a=s.shape;if(r){const t=k(n,e.shape);a=sl(s.shape,t)}return ti(s,a)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yl=pa({euclideanNorm_:function(e,t=null,n=!1){return bl(e,"euclidean",t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wl=pa({exp_:function(e){const t={x:ha(e,"x","exp")};return ta.runKernel(nt,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vl=pa({expandDims_:function(e,t=0){const n=ha(e,"x","expandDims","string_or_numeric");u(t<=n.rank,()=>"Axis must be <= rank of the tensor");const r={input:n},s={dim:t};return ta.runKernel(rt,r,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kl=pa({expm1_:function(e){const t={x:ha(e,"x","expm1")};return ta.runKernel(st,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xl=pa({tile_:function(e,t){const n=ha(e,"x","tile","string_or_numeric");u(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const r={x:n},s={reps:t};return ta.runKernel(Xn,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const El=pa({eye_:function(e,t,n,r="float32"){null==t&&(t=e);const s=wo([e,t],r),a=e<=t?e:t;for(let i=0;i<a;++i)s.set(1,i,i);const o=ti(s.toTensor(),[e,t]);if(null==n)return o;if(1===n.length)return xl(vl(o,0),[n[0],1,1]);if(2===n.length)return xl(vl(vl(o,0),0),[n[0],n[1],1,1]);if(3===n.length)return xl(vl(vl(vl(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sl=pa({floor_:function(e){const t={x:ha(e,"x","floor","float32")};return ta.runKernel(lt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $l=pa({gather_:function(e,t,n=0,r=0){const s={x:ha(e,"x","gather"),indices:ha(t,"indices","gather","int32")},a={axis:n,batchDims:r};return ta.runKernel(ht,s,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Al=pa({greater_:function(e,t){let n=ha(e,"a","greater","string_or_numeric"),r=ha(t,"b","greater","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(ft,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _l=pa({greaterEqual_:function(e,t){let n=ha(e,"a","greaterEqual","string_or_numeric"),r=ha(t,"b","greaterEqual","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(pt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Il=pa({imag_:function(e){const t={input:ha(e,"input","imag")};return ta.runKernel(bt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ml=pa({isFinite_:function(e){const t={x:ha(e,"x","isFinite")};return ta.runKernel(yt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tl=pa({isInf_:function(e){const t={x:ha(e,"x","isInf")};return ta.runKernel(wt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nl=pa({isNaN_:function(e){const t={x:ha(e,"x","isNaN")};return ta.runKernel(vt,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dl=pa({leakyRelu_:function(e,t=.2){const n={x:ha(e,"x","leakyRelu")},r={alpha:t};return ta.runKernel(kt,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rl=pa({less_:function(e,t){let n=ha(e,"a","less","string_or_numeric"),r=ha(t,"b","less","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(xt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bl=pa({lessEqual_:function(e,t){let n=ha(e,"a","lessEqual","string_or_numeric"),r=ha(t,"b","lessEqual","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Et,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fl(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const r={start:e,stop:t,num:n};return ta.runKernel(St,{},r)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cl=pa({localResponseNormalization_:function(e,t=5,n=1,r=1,s=.5){const a=ha(e,"x","localResponseNormalization");u(4===a.rank||3===a.rank,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got\n               rank ${a.rank}.`),u(m(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,i=!1;3===a.rank&&(i=!0,o=ti(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const l={x:o},c={depthRadius:t,bias:n,alpha:r,beta:s},h=ta.runKernel(Tt,l,c);return i?ti(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ol=pa({log_:function(e){const t={x:ha(e,"x","log","float32")};return ta.runKernel($t,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pl=pa({log1p_:function(e){const t={x:ha(e,"x","log1p")};return ta.runKernel(At,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zl(e,t){u(B(e),()=>"The f passed in variableGrads(f) must be a function"),u(null==t||Array.isArray(t)&&t.every(e=>e instanceof Ts),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=null!=t;if(!n){t=[];for(const e in ta.registeredVariables)t.push(ta.registeredVariables[e])}const r=n?t.filter(e=>!e.trainable):null,s=t.length;u((t=t.filter(e=>e.trainable)).length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${s} variables is trainable.`);const{value:a,grads:o}=ta.gradients(e,t,null,!0);u(o.some(e=>null!=e),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),u(0===a.rank,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);const i={};return t.forEach((e,t)=>{null!=o[t]&&(i[e.name]=o[t])}),null!=r&&r.forEach(e=>i[e.name]=null),{value:a,grads:i}}function Ll(e){return ta.customGrad(e)}function Ul(e){if(e.filter(e=>null==e).length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that\n    the f you passed encloses all operations that lead from x to y.")}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wl=pa({neg_:function(e){const t={x:ha(e,"x","neg")};return ta.runKernel(jt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gl=pa({softplus_:function(e){const t={x:ha(e,"x","softplus")};return ta.runKernel(Nn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kl=pa({logSigmoid_:function(e){const t=ha(e,"x","logSigmoid");return Ll(e=>({value:Wl(Gl(Wl(e))),gradFunc:t=>Ao(t,oi(Wl(e)))}))(t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ql=pa({sub_:function(e,t){let n=ha(e,"a","sub"),r=ha(t,"b","sub");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(Jn,s)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jl=pa({logSoftmax_:function(e,t=-1){const n=ha(e,"logits","logSoftmax");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return Ll((e,n)=>{const r=ul(e,t,!0),s=ql(e,r),a=ql(vo(s,"float32"),Ol(gl(wl(s),t,!0)));n([a]);return{value:a,gradFunc:(e,n)=>{const[r]=n,s=wl(r);return ql(e,Ao(gl(e,t,!0),s))}}})(n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vl=pa({logSumExp_:function(e,t=null,n=!1){const r=ha(e,"x","logSumExp"),s=k(t,r.shape),a=ul(r,s,!0),o=ql(r,a),i=wl(o),l=gl(i,s),u=Ol(l),c=Eo(ti(a,u.shape),u);if(n){const e=sl(c.shape,s);return ti(c,e)}return c}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hl=pa({logicalAnd_:function(e,t){const n=ha(e,"a","logicalAnd","bool"),r=ha(t,"b","logicalAnd","bool");Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(_t,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jl=pa({logicalNot_:function(e){const t={x:ha(e,"x","logicalNot","bool")};return ta.runKernel(It,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zl=pa({logicalOr_:function(e,t){const n=ha(e,"a","logicalOr","bool"),r=ha(t,"b","logicalOr","bool");Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Mt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yl=pa({logicalXor_:function(e,t){const n=ha(e,"a","logicalXor","bool"),r=ha(t,"b","logicalXor","bool");return Ki(n.shape,r.shape),Hl(Zl(e,t),Jl(Hl(e,t)))}}),Xl=2147483648;
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ql=pa({searchSorted_:function(e,t,n="left"){const r=ha(e,"sortedSequence","searchSorted"),s=ha(t,"values","searchSorted"),a=r.shape[r.shape.length-1],o=s.shape[s.shape.length-1],i=ti(r,[-1,a]),l=ti(s,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==l.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(d(l.shape)>=Xl)throw new Error("values tensor size must less than 2147483648");if(i.shape[1]>=Xl)throw new Error(`trailing dim_size must less than 2147483648 for int32 output type, was ${i.shape[1]}`);const u={sortedSequence:i,values:l},c={side:n};return ta.runKernel(En,u,c)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eu(e,t){return Ql(e,t,"left")}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tu=pa({maxPool_:function(e,t,n,r,s){const a=ha(e,"x","maxPool");let o=a,i=!1;3===a.rank&&(i=!0,o=ti(a,[1,a.shape[0],a.shape[1],a.shape[2]])),u(4===o.rank,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),u(Yo(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),ei("maxPool",r,s);const l={x:o},c={filterSize:t,strides:n,pad:r,dimRoundingMode:s},h=ta.runKernel(Bt,l,c);return i?ti(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nu=pa({maxPool3d_:function(e,t=[1,1,1],n,r,s,a="NDHWC"){const o=ha(e,"x","maxPool3d");let i=o,l=!1;4===o.rank&&(l=!0,i=ti(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),u(5===i.rank,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),u("NDHWC"===a,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),ei("maxPool3d",r,s);const c={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:a},d=ta.runKernel(Ct,c,h);return l?ti(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ru=pa({maxPoolWithArgmax_:function(e,t,n,r,s=!1){const a={x:ha(e,"x","maxPoolWithArgmax")},o={filterSize:t,strides:n,pad:r,includeBatchInIndex:s},i=ta.runKernel(Pt,a,o);return{result:i[0],indexes:i[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const su=pa({maximum_:function(e,t){let n=ha(e,"a","maximum"),r=ha(t,"b","maximum");[n,r]=js(n,r),"bool"===n.dtype&&(n=vo(n,"int32"),r=vo(r,"int32")),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Rt,s)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const au=pa({mean_:function(e,t=null,n=!1){const r={x:ha(e,"x","mean")},s={axis:t,keepDims:n};return ta.runKernel(zt,r,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ou(e,t="float32"){if(W(e),"complex64"===t){const t=ou(e,"float32"),n=ou(e,"float32");return ga(t,n)}const n=U(d(e),t);return ta.makeTensor(n,e,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iu(e,t="float32"){if(W(e),"complex64"===t){const t=iu(e,"float32"),n=ou(e,"float32");return ga(t,n)}const n=L(d(e),t);return ta.makeTensor(n,e,t)}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lu(e,t,{indexing:n="xy"}={}){if("xy"!==n&&"ij"!==n)throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(void 0===e)return[];let r=ha(e,"x","meshgrid",e instanceof Is?e.dtype:"float32");if(void 0===t)return[r];let s=ha(t,"y","meshgrid",t instanceof Is?t.dtype:"float32");const a=d(r.shape),o=d(s.shape);return"xy"===n?(r=ti(r,[1,-1]),s=ti(s,[-1,1]),[ai(iu([o,1],r.dtype),r),ai(s,iu([1,a],s.dtype))]):(r=ti(r,[-1,1]),s=ti(s,[1,-1]),[ai(r,iu([1,o],r.dtype)),ai(iu([a,1],s.dtype),s)])}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uu=pa({minimum_:function(e,t){let n=ha(e,"a","minimum"),r=ha(t,"b","minimum");[n,r]=js(n,r),"bool"===n.dtype&&(n=vo(n,"int32"),r=vo(r,"int32")),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Ut,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cu=pa({mirrorPad_:function(e,t,n){u("reflect"===n||"symmetric"===n,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const r=ha(e,"x","mirrorPad");if(0===r.rank)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");u(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);const s="reflect"===n?1:0;for(let i=0;i<r.rank;i++)u(2===t[i].length,()=>"Invalid number of paddings. Must be length of 2 each."),u(t[i][0]>=0&&t[i][0]<=r.shape[i]-s&&t[i][1]>=0&&t[i][1]<=r.shape[i]-s,()=>`Padding in dimension ${i} cannot be greater than or equal to ${r.shape[i]-s} or less than 0 for input of shape ${r.shape}`);const a={paddings:t,mode:n},o={x:r};return ta.runKernel(Wt,o,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hu=pa({mod_:function(e,t){let n=ha(e,"a","mod"),r=ha(t,"b","mod");[n,r]=js(n,r);const s={a:n,b:r};return ta.runKernel(Gt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const du=pa({moments_:function(e,t=null,n=!1){const r=k(t,(e=ha(e,"x","moments")).shape),s=au(e,r,n);let a=s.shape;n||(a=sl(s.shape,r));const o=pl(ql(vo(e,"float32"),ti(s,a)));return{mean:s,variance:au(o,r,n)}}});const fu=pa({multiRNNCell_:function(e,t,n,r){const s=ha(t,"data","multiRNNCell"),a=da(n,"c","multiRNNCell"),o=da(r,"h","multiRNNCell");let i=s;const l=[];for(let h=0;h<e.length;h++){const t=e[h](i,a[h],o[h]);l.push(t[0]),l.push(t[1]),i=t[1]}const u=[],c=[];for(let h=0;h<l.length;h+=2)u.push(l[h]),c.push(l[h+1]);return[u,c]}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pu=pa({multinomial_:function(e,t,n,r=!1){const s=ha(e,"logits","multinomial"),a=s.size,o=s.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const i={logits:1===o?ti(s,[1,-1]):s},l={numSamples:t,seed:n,normalized:r},u=ta.runKernel(Kt,i,l);return 1===o?ti(u,[u.size]):u}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gu=pa({notEqual_:function(e,t){let n=ha(e,"a","notEqual","string_or_numeric"),r=ha(t,"b","notEqual","string_or_numeric");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Vt,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mu=pa({oneHot_:function(e,t,n=1,r=0,s="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const a={indices:ha(e,"indices","oneHot","int32")},o={dtype:s,depth:t,onValue:n,offValue:r};return ta.runKernel(Xt,a,o)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bu=pa({onesLike_:function(e){const t={x:ha(e,"x","onesLike")};return ta.runKernel(Yt,t)}});const yu=pa({outerProduct_:function(e,t){const n=ha(e,"v1","outerProduct"),r=ha(t,"v2","outerProduct");u(1===n.rank&&1===r.rank,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`);const s=ti(n,[-1,1]),a=ti(r,[1,-1]);return ai(s,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wu=pa({pad_:function(e,t,n=0){const r=ha(e,"x","pad");if(0===r.rank)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const s={paddings:t,constantValue:n},a={x:r};return ta.runKernel(en,a,s)}});const vu=pa({pad1d_:function(e,t,n=0){return u(2===t.length,()=>"Invalid number of paddings. Must be length of 2."),wu(e,[t],n)}});const ku=pa({pad2d_:function(e,t,n=0){return u(2===t.length&&2===t[0].length&&2===t[1].length,()=>"Invalid number of paddings. Must be length of 2 each."),wu(e,t,n)}});const xu=pa({pad3d_:function(e,t,n=0){return u(3===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length,()=>"Invalid number of paddings. Must be length of 2 each."),wu(e,t,n)}});const Eu=pa({pad4d_:function(e,t,n=0){return u(4===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length&&2===t[3].length,()=>"Invalid number of paddings. Must be length of 2 each."),wu(e,t,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Su=pa({spaceToBatchND_:function(e,t,n){const r=ha(e,"x","spaceToBatchND");u(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),u(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),u(r.shape.reduce((e,r,s)=>s>0&&s<=t.length?e&&(r+n[s-1][0]+n[s-1][1])%t[s-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const s={x:r},a={blockShape:t,paddings:n};return ta.runKernel(Bn,s,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $u=pa({pool_:function(e,t,n,r,s,a,o){null==s&&(s=[1,1]),null==a&&(a=1),0===r&&(r="valid");const i=ha(e,"x","maxPool");let l=i,c=!1;3===i.rank&&(c=!0,l=ti(i,[1,i.shape[0],i.shape[1],i.shape[2]])),u(Yo(a,s),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${s}'`);const h=Uo(l.shape,t,a,s,r),d=[h.dilationHeight,h.dilationWidth];let f;f="same"===r?function(e,t){const n=e.map((e,n)=>e+(e-1)*(t[n]-1)),r=n.map(e=>e-1),s=r.map(e=>Math.floor(e/2)),a=r.map((e,t)=>e-s[t]);return r.map((e,t)=>[s[t],a[t]])}([h.filterHeight,h.filterWidth],d):[[0,0],[0,0]];const p=1===d[0]&&1===d[1],[g,m]=function(e,t,n){const r=n.map(e=>e[0]),s=n.map(e=>e[1]),a=e.concat(r,s),o=t.map((e,t)=>(e-a[t]%e)%e),i=s.map((e,t)=>e+o[t]),l=t.map((e,t)=>[r[t],i[t]]),u=t.map((e,t)=>[0,o[t]]);return[l,u]}([h.inHeight,h.inWidth],d,f),b=p?r:"valid",y=p?l:Su(l,d,g),w=("avg"===n?()=>ni(y,t,a,b,o):()=>tu(y,t,a,b,o))(),v=p?w:ci(w,d,m);return c?ti(v,[v.shape[1],v.shape[2],v.shape[3]]):v}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Au=pa({prelu_:function(e,t){const n={x:ha(e,"x","prelu"),alpha:ha(t,"alpha","prelu")};return ta.runKernel(nn,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _u=pa({prod_:function(e,t=null,n=!1){let r=ha(e,"x","prod");"bool"===r.dtype&&(r=vo(r,"int32"));const s={x:r},a={axis:t,keepDims:n};return ta.runKernel(rn,s,a)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Iu=pa({raggedGather_:function(e,t,n,r){const s={paramsNestedSplits:e.map((e,t)=>ha(e,`tensors${t}`,"raggedGather","int32")),paramsDenseValues:ha(t,"paramsDenseValues","raggedGather"),indices:ha(n,"indices","raggedGather","int32")},a={outputRaggedRank:r},o=ta.runKernel(sn,s,a);return{outputNestedSplits:o.slice(0,o.length-1),outputDenseValues:o[o.length-1]}}});
/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mu=pa({raggedRange_:function(e,t,n){const r=ha(e,"starts","raggedRange"),s={starts:r,limits:ha(t,"limits","raggedRange",r.dtype),deltas:ha(n,"deltas","raggedRange",r.dtype)},a=ta.runKernel(an,s);return{rtNestedSplits:a[0],rtDenseValues:a[1]}}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tu=pa({raggedTensorToTensor_:function(e,t,n,r,s){const a=ha(e,"shape","raggedTensorToTensor","int32"),o=ha(t,"values","raggedTensorToTensor"),i={shape:a,values:o,defaultValue:ha(n,"defaultValue","raggedTensorToTensor",o.dtype),rowPartitionTensors:r.map((e,t)=>ha(e,`tensors${t}`,"raggedTensorToTensor","int32"))},l={rowPartitionTypes:s};return ta.runKernel(on,i,l)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nu=pa({rand_:function(e,t,n){W(e);const r=d(e);let s=null;if(null==n||"float32"===n)s=new Float32Array(r);else if("int32"===n)s=new Int32Array(r);else{if("bool"!==n)throw new Error(`Unknown data type ${n}`);s=new Uint8Array(r)}for(let a=0;a<r;a++)s[a]=t();return ta.makeTensor(s,e,n)}});var Du={exports:{}};!function(e,t){function n(e){var t=this,n=function(){var e=4022871197,t=function(t){t=String(t);for(var n=0;n<t.length;n++){var r=.02519603282416938*(e+=t.charCodeAt(n));r-=e=r>>>0,e=(r*=e)>>>0,e+=4294967296*(r-=e)}return 2.3283064365386963e-10*(e>>>0)};return t}();t.next=function(){var e=2091639*t.s0+2.3283064365386963e-10*t.c;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=0|e)},t.c=1,t.s0=n(" "),t.s1=n(" "),t.s2=n(" "),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function r(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function s(e,t){var s=new n(e),a=t&&t.state,o=s.next;return o.int32=function(){return 4294967296*s.next()|0},o.double=function(){return o()+11102230246251565e-32*(2097152*o()|0)},o.quick=o,a&&("object"==typeof a&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.alea=s}(0,Du);var Ru=Du.exports,Bu={exports:{}};!function(e,t){function n(e){var t=this,n="";t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),t.next()}function r(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function s(e,t){var s=new n(e),a=t&&t.state,o=function(){return(s.next()>>>0)/4294967296};return o.double=function(){do{var e=((s.next()>>>11)+(s.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=s.next,o.quick=o,a&&("object"==typeof a&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.xor128=s}(0,Bu);var Fu=Bu.exports,Cu={exports:{}};!function(e,t){function n(e){var t=this,n="";t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^e^e<<1)|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function r(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function s(e,t){var s=new n(e),a=t&&t.state,o=function(){return(s.next()>>>0)/4294967296};return o.double=function(){do{var e=((s.next()>>>11)+(s.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=s.next,o.quick=o,a&&("object"==typeof a&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.xorwow=s}(0,Cu);var Ou=Cu.exports,Pu={exports:{}};!function(e,t){function n(e){var t=this;t.next=function(){var e,n,r=t.x,s=t.i;return e=r[s],n=(e^=e>>>7)^e<<24,n^=(e=r[s+1&7])^e>>>10,n^=(e=r[s+3&7])^e>>>3,n^=(e=r[s+4&7])^e<<7,e=r[s+7&7],n^=(e^=e<<13)^e<<9,r[s]=n,t.i=s+1&7,n},function(e,t){var n,r=[];if(t===(0|t))r[0]=t;else for(t=""+t,n=0;n<t.length;++n)r[7&n]=r[7&n]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&0===r[n];++n);for(8==n?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}(t,e)}function r(e,t){return t.x=e.x.slice(),t.i=e.i,t}function s(e,t){null==e&&(e=+new Date);var s=new n(e),a=t&&t.state,o=function(){return(s.next()>>>0)/4294967296};return o.double=function(){do{var e=((s.next()>>>11)+(s.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=s.next,o.quick=o,a&&(a.x&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.xorshift7=s}(0,Pu);var zu=Pu.exports,Lu={exports:{}};!function(e,t){function n(e){var t=this;t.next=function(){var e,n,r=t.w,s=t.X,a=t.i;return t.w=r=r+1640531527|0,n=s[a+34&127],e=s[a=a+1&127],n^=n<<13,e^=e<<17,n^=n>>>15,e^=e>>>12,n=s[a]=n^e,t.i=a,n+(r^r>>>16)|0},function(e,t){var n,r,s,a,o,i=[],l=128;for(t===(0|t)?(r=t,t=null):(t+="\0",r=0,l=Math.max(l,t.length)),s=0,a=-32;a<l;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),0===a&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,s=0==(n=i[127&a]^=r+o)?s+1:0);for(s>=128&&(i[127&(t&&t.length||0)]=-1),s=127,a=512;a>0;--a)r=i[s+34&127],n=i[s=s+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,i[s]=r^n;e.w=o,e.X=i,e.i=s}(t,e)}function r(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function s(e,t){null==e&&(e=+new Date);var s=new n(e),a=t&&t.state,o=function(){return(s.next()>>>0)/4294967296};return o.double=function(){do{var e=((s.next()>>>11)+(s.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=s.next,o.quick=o,a&&(a.X&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.xor4096=s}(0,Lu);var Uu=Lu.exports,Wu={exports:{}};!function(e,t){function n(e){var t=this,n="";t.next=function(){var e=t.b,n=t.c,r=t.d,s=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^s,s=s-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^s,t.a=s-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=0|e):n+=e;for(var r=0;r<n.length+20;r++)t.b^=0|n.charCodeAt(r),t.next()}function r(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function s(e,t){var s=new n(e),a=t&&t.state,o=function(){return(s.next()>>>0)/4294967296};return o.double=function(){do{var e=((s.next()>>>11)+(s.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=s.next,o.quick=o,a&&("object"==typeof a&&r(a,s),o.state=function(){return r(s,{})}),o}t&&t.exports?t.exports=s:this.tychei=s}(0,Wu);var Gu=Wu.exports,Ku={exports:{}};const qu=Er(Object.freeze(Object.defineProperty({__proto__:null,default:{}},Symbol.toStringTag,{value:"Module"})));var ju;ju=Ku,function(e,t,n){var r,s=256,a="random",o=n.pow(s,6),i=n.pow(2,52),l=2*i,u=255;function c(u,c,m){var b=[],y=p(f((c=1==c?{entropy:!0}:c||{}).entropy?[u,g(t)]:null==u?function(){try{var n;return r&&(n=r.randomBytes)?n=n(s):(n=new Uint8Array(s),(e.crypto||e.msCrypto).getRandomValues(n)),g(n)}catch(Lp){var a=e.navigator,o=a&&a.plugins;return[+new Date,e,o,e.screen,g(t)]}}():u,3),b),w=new h(b),v=function(){for(var e=w.g(6),t=o,n=0;e<i;)e=(e+n)*s,t*=s,n=w.g(1);for(;e>=l;)e/=2,t/=2,n>>>=1;return(e+n)/t};return v.int32=function(){return 0|w.g(4)},v.quick=function(){return w.g(4)/4294967296},v.double=v,p(g(w.S),t),(c.pass||m||function(e,t,r,s){return s&&(s.S&&d(s,w),e.state=function(){return d(w,{})}),r?(n[a]=e,t):e})(v,y,"global"in c?c.global:this==n,c.state)}function h(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,i=r.S=[];for(n||(e=[n++]);a<s;)i[a]=a++;for(a=0;a<s;a++)i[a]=i[o=u&o+e[a%n]+(t=i[a])],i[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,i=r.S;e--;)t=i[a=u&a+1],n=n*s+i[u&(i[a]=i[o=u&o+t])+(i[o]=t)];return r.i=a,r.j=o,n})(s)}function d(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function f(e,t){var n,r=[],s=typeof e;if(t&&"object"==s)for(n in e)try{r.push(f(e[n],t-1))}catch(Lp){}return r.length?r:"string"==s?e:e+"\0"}function p(e,t){for(var n,r=e+"",s=0;s<r.length;)t[u&s]=u&(n^=19*t[u&s])+r.charCodeAt(s++);return g(t)}function g(e){return String.fromCharCode.apply(0,e)}if(p(n.random(),t),ju.exports){ju.exports=c;try{r=qu}catch(m){}}else n["seed"+a]=c}("undefined"!=typeof self?self:kr,[],Math);var Vu=Ru,Hu=Fu,Ju=Ou,Zu=zu,Yu=Uu,Xu=Gu,Qu=Ku.exports;Qu.alea=Vu,Qu.xor128=Hu,Qu.xorwow=Ju,Qu.xorshift7=Zu,Qu.xor4096=Yu,Qu.tychei=Xu;var ec=Qu;
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tc(){return 32===ta.backend.floatPrecision()?.001:.1}function nc(e,t,n){let r=!0;if((ps(e)||ps(t))&&(r=!1),ps(e)&&ps(t)&&(r=!0),r){const n=e.constructor.name,r=t.constructor.name;if(n!==r)throw new Error(`Arrays are of different type. Actual: ${n}. Expected: ${r}`)}if(Array.isArray(e)&&Array.isArray(t)){const n=la(e),r=la(t);if(!g(n,r))throw new Error(`Arrays have different shapes. Actual: [${n}]. Expected: [${r}]`)}const s=ps(e)?e:gs(e),a=ps(t)?t:gs(t);if(s.length!==a.length)throw new Error(`Arrays have different lengths actual: ${s.length} vs expected: ${a.length}.\nActual:   ${s}.\nExpected: ${a}.`);for(let o=0;o<a.length;++o){const e=s[o],t=a[o];if(!n(e,t))throw new Error(`Arrays differ: actual[${o}] = ${e}, expected[${o}] = ${t}.\nActual:   ${s}.\nExpected: ${a}.`)}"undefined"!=typeof expect&&expect().nothing()}function rc(e,t,n){return!isFinite(e)&&!isFinite(t)||!(isNaN(e)||isNaN(t)||Math.abs(e-t)>n)}const sc=Object.freeze(Object.defineProperty({__proto__:null,TEST_EPSILON_FLOAT16:.1,createVideoElement:function(e){const t=document.createElement("video");return"playsInline"in t&&(t.playsInline=!0),t.muted=!0,t.loop=!0,t.style.position="fixed",t.style.left="0px",t.style.top="0px",t.preload="auto",t.appendChild(e),new Promise(e=>{t.addEventListener("loadeddata",n=>e(t)),t.load()})},encodeStrings:function e(t){for(let n=0;n<t.length;n++){const r=t[n];Array.isArray(r)?e(r):t[n]=ds(r)}return t},expectArrayBuffersEqual:function(e,t){const n=new Float32Array(e),r=new Float32Array(t);if(n.length!==r.length)throw new Error(`Expected ArrayBuffer to be of length ${r.length}, but it was ${n.length}`);for(let s=0;s<r.length;s++)if(n[s]!==r[s])throw new Error(`Expected ArrayBuffer value at ${s} to be ${r[s]} but got ${n[s]} instead`)},expectArraysClose:function(e,t,n){return null==n&&(n=tc()),nc(e,t,(e,t)=>rc(e,t,n))},expectArraysEqual:function(e,t){const n="string"==typeof t||"number"==typeof t||"boolean"==typeof t?[t]:t;return T(e)||T(e[0])||T(t)||T(t[0])?nc(e,n,(e,t)=>e==t):nc(e,t,(e,t)=>rc(e,t,0))},expectNumbersClose:function(e,t,n){if(null==n&&(n=tc()),!rc(e,t,n))throw new Error(`Numbers differ: actual === ${e}, expected === ${t}`);"undefined"!=typeof expect&&expect().nothing()},expectPromiseToFail:function(e,t){e().then(()=>t.fail(),()=>t()),"undefined"!=typeof expect&&expect().nothing()},expectValuesInRange:function(e,t,n){for(let r=0;r<e.length;r++)if(e[r]<t||e[r]>n)throw new Error(`Value out of range:${e[r]} low: ${t}, high: ${n}`)},play:async function(e){await e.play(),"requestVideoFrameCallback"in e&&await new Promise(t=>{e.requestVideoFrameCallback(t)})},testEpsilon:tc},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ac{constructor(e,t,n,r,s){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+2*this.stdDev,this.lower=this.mean-2*this.stdDev);const a=s||Math.random();this.random=ec.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){const e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,s,a;do{r=2*this.random()-1,s=2*this.random()-1,a=r*r+s*s}while(a>=1||0===a);const o=Math.sqrt(-2*Math.log(a)/a);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*s*o,this.truncated&&!this.isValidTruncated(e)||(n=!0)}return this.truncated&&!this.isValidTruncated(t)||(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return null==this.dtype||"float32"===this.dtype?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class oc{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;const s=r||Math.random();this.randu=ec.alea(s.toString()),this.randn=new ac(0,1,n,!1,this.randu()),this.d=e<1?e+2/3:e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,s,a;for(;;){do{r=this.randn.nextValue(),a=1+this.c*r}while(a<=0);if(a*=a*a,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-a+Math.log(a)),s=this.randu(),s<t||Math.log(s)<n)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(a)}convertValue(e){return"float32"===this.dtype?e:Math.round(e)}}class ic{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>null==this.dtype||"float32"===this.dtype,this.min=e,this.range=t-e,this.dtype=n,null==r&&(r=Math.random()),"number"==typeof r&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=ec.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lc=pa({randomGamma_:function(e,t,n=1,r="float32",s){if(W(e),null==n&&(n=1),null==r&&(r="float32"),"float32"!==r&&"int32"!==r)throw new Error(`Unsupported data type ${r}`);const a=new oc(t,n,r,s),o=wo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uc=pa({randomNormal_:function(e,t=0,n=1,r,s){if(W(e),null!=r&&"bool"===r)throw new Error(`Unsupported data type ${r}`);const a=new ac(t,n,r,!1,s),o=wo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cc=pa({randomStandardNormal_:function(e,t,n){if(null!=t&&"bool"===t)throw new Error(`Unsupported data type ${t}`);return uc(e,0,1,t,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hc=pa({randomUniform_:function(e,t=0,n=1,r="float32",s){W(e);const a=wo(e,r),o=new ic(t,n,null,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dc=pa({randomUniformInt_:function(e,t,n,r){return hc(e,t,n,"int32",r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fc(e,t,n=1,r="float32"){if(0===n)throw new Error("Cannot have a step of zero");const s={start:e,stop:t,step:n,dtype:r};return ta.runKernel(ln,{},s)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pc=pa({real_:function(e){const t={input:ha(e,"input","real")};return ta.runKernel(un,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gc=pa({reciprocal_:function(e){const t={x:ha(e,"x","reciprocal")};return ta.runKernel(cn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mc=pa({relu_:function(e){const t={x:ha(e,"x","relu")};return ta.runKernel(hn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bc=pa({relu6_:function(e){const t={x:ha(e,"x","relu6")};return ta.runKernel(bn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yc=pa({reverse_:function(e,t){const n={x:ha(e,"x","reverse")},r={dims:t};return ta.runKernel(yn,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wc=pa({reverse1d_:function(e){const t=ha(e,"x","reverse");return u(1===t.rank,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),yc(t,0)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vc=pa({reverse2d_:function(e,t){const n=ha(e,"x","reverse");return u(2===n.rank,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),yc(n,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kc=pa({reverse3d_:function(e,t){const n=ha(e,"x","reverse");return u(3===n.rank,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),yc(n,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xc=pa({reverse4d_:function(e,t){const n=ha(e,"x","reverse");return u(4===n.rank,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),yc(n,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ec=pa({round_:function(e){const t={x:ha(e,"x","round")};return ta.runKernel(wn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sc=pa({rsqrt_:function(e){const t={x:ha(e,"x","rsqrt","float32")};return ta.runKernel(vn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $c=pa({selu_:function(e){const t={x:ha(e,"x","selu")};return ta.runKernel($n,t)}});const Ac=pa({separableConv2d_:function(e,t,n,r,s,a=[1,1],o="NHWC"){const i=ha(e,"x","separableConv2d"),l=ha(t,"depthwiseFilter","separableConv2d"),c=ha(n,"pointwiseFilter","separableConv2d");let h=i,d=!1;if(3===i.rank&&(d=!0,h=ti(i,[1,i.shape[0],i.shape[1],i.shape[2]])),"NCHW"===o)throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");u(4===h.rank,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),u(4===l.rank,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),u(4===c.rank,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),u(1===c.shape[0],()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),u(1===c.shape[1],()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const f=l.shape[2],p=l.shape[3];u(c.shape[2]===f*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*p}, but got ${c.shape[2]}.`);const g=zi(h,l,r,s,o,a),m=Ai(g,c,1,"valid",o);return d?ti(m,[m.shape[1],m.shape[2],m.shape[3]]):m}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _c=async function(e,t){const n=ha(e,"x","setdiff1d"),r=ha(t,"y","setdiff1d");u(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),u(1===n.rank,()=>`x should be 1D tensor, but got x (${n.shape}).`),u(1===r.rank,()=>`y should be 1D tensor, but got y (${r.shape}).`);const s=await n.data(),a=await r.data(),o=new Set(a);let i=0;for(let u=0;u<s.length;u++)o.has(s[u])||i++;const l=new $s([i],n.dtype),c=new $s([i],"int32");for(let u=0,h=0;u<s.length;u++)o.has(s[u])||(l.values[h]=s[u],c.values[h]=u,h++);return[l.toTensor(),c.toTensor()]};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ic=pa({sign_:function(e){const t={x:ha(e,"x","sign")};return ta.runKernel(Mn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mc=pa({sin_:function(e){const t={x:ha(e,"x","sin","float32")};return ta.runKernel(_n,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tc=pa({sinh_:function(e){const t={x:ha(e,"x","sinh")};return ta.runKernel(In,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nc=pa({slice1d_:function(e,t,n){const r=ha(e,"x","slice1d");return u(1===r.rank,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),ii(r,[t],[n])}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dc=pa({slice2d_:function(e,t,n){const r=ha(e,"x","slice2d");return u(2===r.rank,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),ii(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rc=pa({slice3d_:function(e,t,n){const r=ha(e,"x","slice3d");return u(3===r.rank,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),ii(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bc=pa({slice4d_:function(e,t,n){const r=ha(e,"x","slice4d");return u(4===r.rank,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),ii(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fc=pa({softmax_:function(e,t=-1){const n=ha(e,"logits","softmax","float32");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const r={logits:n},s={dim:t};return ta.runKernel(Cn,r,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cc=pa({fft_:function(e){u("complex64"===e.dtype,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return ta.runKernel(at,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oc=pa({ifft_:function(e){u("complex64"===e.dtype,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return ta.runKernel(mt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pc=pa({irfft_:function(e){const t=e.shape[e.shape.length-1],n=e.size/t;let r;if(t<=2){const s=ti(e,[n,t]);r=Oc(s)}else{const s=[n,2*(t-1)],a=ti(pc(e),[n,t]),o=ti(Il(e),[n,t]),i=yc(ii(a,[0,1],[n,t-2]),1),l=Ao(yc(ii(o,[0,1],[n,t-2]),1),dl(-1)),u=si([a,i],1),c=si([o,l],1),h=ti(ga(u,c),[s[0],s[1]]);r=Oc(h)}if(r=pc(r),3===e.rank&&0!==e.shape[0]){const t=r,n=e.shape[0];r=ti(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zc=pa({split_:function(e,t,n=0){const r={x:ha(e,"x","split")},s={numOrSizeSplits:t,axis:n};return ta.runKernel(Fn,r,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lc=pa({rfft_:function(e,t){u("float32"===e.dtype,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const r=e.size/n;let s;if(null!=t&&t<n){const r=e.shape.map(e=>0),a=e.shape.map(e=>e);a[e.shape.length-1]=t,s=ii(e,r,a),n=t}else if(null!=t&&t>n){const r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,s=si([e,ou(r)],e.shape.length-1),n=t}else s=e;const a=Hi(s),o=ti(ga(s,a),[r,n]),i=Cc(o),l=Math.floor(n/2)+1,c=pc(i),h=Il(i),d=zc(c,[l,n-l],c.shape.length-1),f=zc(h,[l,n-l],h.shape.length-1),p=s.shape.slice();return p[s.shape.length-1]=l,ti(ga(d[0],f[0]),p)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uc=pa({squaredDifference_:function(e,t){let n=ha(e,"a","squaredDifference"),r=ha(t,"b","squaredDifference");[n,r]=js(n,r),Ki(n.shape,r.shape);const s={a:n,b:r};return ta.runKernel(Wn,s,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wc=pa({squeeze_:function(e,t){const n=ha(e,"x","squeeze","string_or_numeric");return ti(n,x(n.shape,t).newShape)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gc=pa({stack_:function(e,t=0){const n=da(e,"tensors","stack","string_or_numeric");u(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&u(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const r=n,s={axis:t};return ta.runKernel(Qt,r,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kc=pa({step_:function(e,t=0){const n={x:ha(e,"x","step")},r={alpha:t};return ta.runKernel(or,n,r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qc=pa({stridedSlice_:function(e,t,n,r,s=0,a=0,o=0,i=0,l=0){const u={x:ha(e,"x","stridedSlice","string_or_numeric")},c={begin:t,end:n,strides:r,beginMask:s,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:l};return ta.runKernel(qn,u,c)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jc=pa({tan_:function(e){const t={x:ha(e,"x","tan","float32")};return ta.runKernel(Zn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vc(e,t){h(e);const n=la(e,t);if(1!==n.length)throw new Error("tensor1d() requires values to be a flat/TypedArray");return ma(e,null,n,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hc(e,t,n){if(h(e),null!=t&&2!==t.length)throw new Error("tensor2d() requires shape to have two numbers");const r=la(e,n);if(2!==r.length&&1!==r.length)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return ma(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jc(e,t,n){if(h(e),null!=t&&3!==t.length)throw new Error("tensor3d() requires shape to have three numbers");const r=la(e,n);if(3!==r.length&&1!==r.length)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return ma(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zc(e,t,n){if(h(e),null!=t&&4!==t.length)throw new Error("tensor4d() requires shape to have four numbers");const r=la(e,n);if(4!==r.length&&1!==r.length)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return ma(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yc(e,t,n){if(h(e),null!=t&&5!==t.length)throw new Error("tensor5d() requires shape to have five numbers");const r=la(e,n);if(5!==r.length&&1!==r.length)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return ma(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xc(e,t,n){if(h(e),null!=t&&6!==t.length)throw new Error("tensor6d() requires shape to have six numbers");const r=la(e,n);if(6!==r.length&&1!==r.length)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return ma(e,t=t||r,r,n)}function Qc(e,t,n){const r=t.rank>1?t.shape[t.rank-1]:1,s=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${s}.`;if(n.rank<s)throw new Error(a+` update.rank < ${s}. `);if(e.length<r+(n.rank-s))throw new Error(a+` Output shape length < ${r+(n.rank-s)}`);if(n.rank!==s+e.length-r)throw new Error(a+" update.rank != "+(s+e.length-r));for(let o=0;o<s;++o)if(n.shape[o]!==t.shape[o])throw new Error(a+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${t.shape[o]}).`);for(let o=0;o<n.rank-s;++o)if(n.shape[o+s]!==e[o+r])throw new Error(a+` updates.shape[${o+s}] (${n.shape[o+s]}) != shape[${o+s}] (${e[o+s]})`)}function eh(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if("int32"!==t.dtype)throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(0===n.length){if(0===t.size)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(0===e.size)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Qc(n,t,e)}function th(e,t,n){const r=t.shape.length,s=r>1?t.shape[r-1]:1,a=n.length;let o=1;for(let l=s;l<a;++l)o*=n[l];const i=s<1?1:s;return{sliceRank:s,numUpdates:d(t.shape)/i,sliceSize:o,strides:[...C(n.slice(0,s)),1],outputSize:d(n)}}const nh=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:th,validateInput:eh,validateUpdateShape:Qc},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rh=pa({tensorScatterUpdate_:function(e,t,n){const r=ha(e,"tensor","tensorScatterupdate"),s=ha(t,"indices","tensorScatterupdate","int32"),a=ha(n,"updates","tensorScatterupdate");if(eh(a,s,r.shape),r.dtype!==a.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${r.dtype} and ${a.dtype}.`);const o={tensor:r,indices:s,updates:a};return ta.runKernel(xn,o,{})}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sh=pa({topk_:function(e,t=1,n=!0){const r=ha(e,"x","topk");if(0===r.rank)throw new Error("topk() expects the input to be of rank 1 or higher");const s=r.shape[r.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>s)throw new Error(`'k' passed to topk() must be <= the last dimension (${s}) but got ${t}`);const a={x:r},o={k:t,sorted:n},[i,l]=ta.runKernel(Qn,a,o);return{values:i,indices:l}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ah=pa({truncatedNormal_:function(e,t=0,n=1,r,s){if(W(e),null!=r&&"bool"===r)throw new Error("Unsupported data type $ { dtype }");const a=new ac(t,n,r,!0,s),o=wo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oh=pa({unique_:function(e,t=0){const n=ha(e,"x","unique","string_or_numeric");u(n.rank>0,()=>"The input tensor must be at least 1D");const r={x:n},s={axis:t},[a,o]=ta.runKernel(nr,r,s);return{values:a,indices:o}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ih=pa({unsortedSegmentSum_:function(e,t,n){const r=ha(e,"x","unsortedSegmentSum"),s=ha(t,"segmentIds","unsortedSegmentSum","int32");u(m(n),()=>"numSegments must be of dtype int");const a={x:r,segmentIds:s},o={numSegments:n};return ta.runKernel(sr,a,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lh=pa({unstack_:function(e,t=0){const n=ha(e,"x","unstack","string_or_numeric");u(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const r={value:n},s={axis:t};return ta.runKernel(rr,r,s)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uh(e,t){return Ql(e,t,"right")}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ch(e,t=!0,n,r){return ta.makeVariable(e,t,n,r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hh(e,t){const n=[];for(let a=0;a<t.length;a++)t[a]&&n.push(a);const r=wo(e,"int32"),s=wo([n.length,e.length],"int32");for(let a=0;a<n.length;a++){const t=r.indexToLoc(n[a]),o=a*e.length;s.values.set(t,o)}return s.toTensor()}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dh=async function(e){const t=ha(e,"condition","whereAsync","bool"),n=await t.data(),r=hh(t.shape,n);return e!==t&&t.dispose(),r};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fh=async function(e,t,n){const r=ha(e,"tensor","boolMask"),s=ha(t,"mask","boolMask","bool"),a=null==n?0:n,o=s.rank,i=r.shape;u(o>0,()=>"mask cannot be scalar"),c(i.slice(a,a+o),s.shape,"mask's shape must match the first K dimensions of tensor's shape,");let l=1;for(let u=a;u<a+o;u++)l*=i[u];const h=i.slice(0,a).concat([l],i.slice(a+o)),d=ti(r,h),f=ti(s,[-1]),p=await dh(f),g=Wc(p,[1]),m=$l(d,g,a);return e!==r&&r.dispose(),t!==s&&s.dispose(),g.dispose(),d.dispose(),f.dispose(),p.dispose(),m};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ph=pa({transpose_:function(e,t,n){const r=ha(e,"x","transpose");if(null==t&&(t=r.shape.map((e,t)=>t).reverse()),u(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{u(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();const s={x:r},a={perm:t};return"complex64"===r.dtype?ka(()=>{let e=pc(r),t=Il(r);return e=ta.runKernel(tr,{x:e},a),t=ta.runKernel(tr,{x:t},a),n&&(t=Wl(t)),ga(e,t)}):ta.runKernel(tr,s,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gh=pa({movingAverage_:function(e,t,n,r,s=!0){const a=ha(e,"v","movingAverage"),o=ha(t,"x","movingAverage"),i=ha(n,"decay","movingAverage");Vs(a,o),u(g(a.shape,o.shape),()=>"Shape mismatch in v and x");const l=dl(1),c=ql(l,i);let h=Ao(ql(o,a),c);if(s){u(null!=r,()=>"When using zeroDebias: true, step is required.");const e=ha(r,"step","movingAverage");h=$o(h,ql(l,hl(i,e)))}return Eo(a,h)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mh=pa({scatterND_:function(e,t,n){W(n);const r=ha(e,"indices","scatterND","int32"),s=ha(t,"updates","scatterND");eh(s,r,n);const a={indices:r,updates:s},o={shape:n};return ta.runKernel(kn,a,o)}});const bh=pa({sparseToDense_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r=0){W(n);const s=ha(e,"sparseIndices","sparseToDense","int32"),a=ha(t,"sparseValues","sparseToDense","string_or_numeric"),o=ha(r,"defaultValue","sparseToDense",a.dtype);!function(e,t,n,r){if("int32"!==e.dtype)throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const s=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);const o=t.size;if(0!==t.rank&&(1!==t.rank||o!==s))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${s}]`);if(t.dtype!==r.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}(s,a,n,o);const i={sparseIndices:s,sparseValues:a,defaultValue:o},l={outputShape:n};return ta.runKernel(Un,i,l)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yh=pa({gatherND_:function(e,t){const n=ha(t,"indices","gatherND","int32"),r={params:ha(e,"x","gatherND","string_or_numeric"),indices:n};return ta.runKernel(dt,r)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wh=pa({dropout_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r){const s=ha(e,"x","dropout");if(u("float32"===s.dtype,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${s.dtype} tensor instead.`),u(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),0===t)return e instanceof Is?s.clone():s;const a=function(e,t){if(null==t)return e.shape.slice();if(g(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let r=0;r<e.shape.length;r++)null==t[r]&&null!=e.shape[r]?n.push(e.shape[r]):n.push(t[r]);return n}return t}(s,n),o=1-t,i=$o(Sl(Eo(hc(a,0,1,"float32",r),o)),o);return Ao(s,i)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vh(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function kh(e,t,n){const r=1-e%2,s=new Float32Array(e);for(let a=0;a<e;++a){const o=2*Math.PI*a/(e+r-1);s[a]=t-n*Math.cos(o)}return Vc(s,"float32")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xh=async function(e,t,n=1){const r=ha(e,"predictions","inTopK"),s=ha(t,"targets","inTopK");u(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),u(r.rank-1===s.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${s.rank}`),c(r.shape.slice(0,r.shape.length-1),s.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=r.shape[r.shape.length-1];u(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);const o=await r.data(),i=await s.data(),[l,h]=[o.length/a,a],d=E("bool",l);for(let u=0;u<l;u++){const e=u*h,t=o.subarray(e,e+h),r=[];for(let n=0;n<t.length;n++)r.push({value:t[n],index:n});r.sort((e,t)=>t.value-e.value),d[u]=0;for(let s=0;s<n;s++)if(r[s].index===i[u]){d[u]=1;break}}return e!==r&&r.dispose(),t!==s&&s.dispose(),ba(d,s.shape,"bool")};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Eh=pa({conv2DBackpropFilter_:function(e,t,n,r,s,a="NHWC",o){let i=e;3===e.rank&&(i=ti(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let l=t;3===l.rank&&(l=ti(t,[1,t.shape[0],t.shape[1],t.shape[2]])),u(4===i.rank,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),u(4===l.rank,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${l.shape}.`),u(4===n.length,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const c="NHWC"===a?i.shape[3]:i.shape[1],h="NHWC"===a?l.shape[3]:l.shape[1];u(c===n[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${n[2]}.`),u(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),ei("conv2dDerFilter",s,o);const d={x:i,dy:l},f={strides:r,pad:s,dataFormat:a,dimRoundingMode:o,filterShape:n};return ta.runKernel(Te,d,f)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sh(e,t,n){if(null==n||"linear"===n)return e;if("relu"===n)return Ao(e,Kc(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function $h(e,t){let n=t;const r=Gi(e.shape,t.shape);return r.length>0&&(n=gl(n,r)),ti(n,e.shape)}function Ah(e,t,n,r){if("linear"===t)return e;if("relu"===t)return mc(e);if("elu"===t)return Xi(e);if("relu6"===t)return bc(e);if("prelu"===t)return Au(e,n);if("leakyrelu"===t)return Dl(e,r);if("sigmoid"===t)return oi(e);throw new Error(`Unknown fused activation ${t}.`)}const _h=(e,t)=>!(e>0)||"linear"===t;
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ih=pa({fusedConv2d_:function({x:e,filter:t,strides:n,pad:r,dataFormat:s="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:l="linear",preluActivationWeights:c,leakyreluAlpha:h}){if(l=l||"linear",!1===_h(ta.state.gradientDepth,l)){u("NHWC"===s,()=>`Error in fused conv2d: got dataFormat of ${s} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let d=Ai(e,t,n,r,s,a,o);return null!=i&&(d=Eo(d,i)),Ah(d,l,c,h)}const d=ha(e,"x","conv2d","float32"),f=ha(t,"filter","conv2d","float32");let p=d,g=!1;3===d.rank&&(g=!0,p=ti(d,[1,d.shape[0],d.shape[1],d.shape[2]])),u(4===p.rank,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),u(4===f.rank,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),ei("fused conv2d",r,o);const m="NHWC"===s?p.shape[3]:p.shape[1];u(f.shape[2]===m,()=>`Error in conv2d: depth of input (${m}) must match input depth for filter ${f.shape[2]}.`),u(Yo(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const b=Go(p.shape,f.shape,n,a,r,o);let y,w;if(null!=i&&(y=ha(i,"bias","fused conv2d"),[y]=js(y,d),"NHWC"===s?Ki(b.outShape,y.shape):(u(y.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${y.shape.length}.`),u(0===y.shape.length||y.shape[0]===b.outChannels||1===y.shape[0],()=>`Error in fused conv2d: bias shape (${y.shape}) is not compatible with the number of output channels (${b.outChannels})`))),null!=c){const e=c.shape;if(u(e.length<=1||3===e.length,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),1===e.length)u(1===e[0]||e[0]===b.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${b.outChannels}).`);else if(3===e.length)try{Ki(e,b.outShape)}catch(Lp){const n=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${b.outShape}).`;throw Error(n)}w=ha(c,"prelu weights","fused conv2d")}const v=(e,t)=>{u("NHWC"===s,()=>`Error in gradient of fused conv2D: got dataFormat of ${s} but only NHWC is currently supported.`);const[o,i,c,h]=t,d=Sh(e,c,l);u(Zo(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const f=[Ii(i.shape,d,o,n,r),Eh(i,d,o.shape,n,r)];if(null!=h){const e=$h(h,d);f.push(e)}return f},k={x:p,filter:f,bias:y,preluActivationWeights:w},x={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o,activation:l,leakyreluAlpha:h};if(null==i){return Ll((e,t,n)=>{let r=ta.runKernel(cr,k,x);return n([t,e,r]),g&&(r=ti(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:v}})(p,f)}return Ll((e,t,n,r)=>{let s=ta.runKernel(cr,k,x);return r([t,e,s,n]),g&&(s=ti(s,[s.shape[1],s.shape[2],s.shape[3]])),{value:s,gradFunc:v}})(p,f,y)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mh=pa({depthwiseConv2dNativeBackpropFilter_:function(e,t,n,r,s,a=[1,1],o){let i=e;3===e.rank&&(i=ti(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let l=t;3===l.rank&&(l=ti(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={x:i,dy:l},c={strides:r,pad:s,dimRoundingMode:o,dilations:a,filterShape:n};return ta.runKernel(Ge,u,c)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Th=pa({depthwiseConv2dNativeBackpropInput_:function(e,t,n,r,s,a=[1,1],o){let i=t,l=!1;3===t.rank&&(l=!0,i=ti(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={dy:i,filter:n},c={strides:r,pad:s,dimRoundingMode:o,dilations:a,inputShape:e},h=ta.runKernel(Ke,u,c);return l?ti(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nh=pa({fusedDepthwiseConv2d_:function({x:e,filter:t,strides:n,pad:r,dataFormat:s="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:l="linear",preluActivationWeights:c,leakyreluAlpha:h}){if(!1===_h(ta.state.gradientDepth,l)){let u=zi(e,t,n,r,s,a,o);return null!=i&&(u=Eo(u,i)),Ah(u,l,c,h)}const d=ha(e,"x","depthwiseConv2d","float32"),f=ha(t,"filter","depthwiseConv2d","float32");let p=d,g=!1;3===d.rank&&(g=!0,p=ti(d,[1,d.shape[0],d.shape[1],d.shape[2]])),u(4===p.rank,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${p.rank}.`),u(4===f.rank,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),u(p.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${p.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),null==a&&(a=[1,1]),u(Yo(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),ei("fused depthwiseConv2d",r,o);const m=Go(p.shape,f.shape,n,a,r,o,!0);let b,y;null!=i&&(b=ha(i,"bias","fused conv2d"),[b]=js(b,d),Ki(m.outShape,b.shape)),null!=c&&(y=ha(c,"prelu weights","fused depthwiseConv2d"));const w=(e,t)=>{u(Zo(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[s,i,c,h]=t,d=Sh(e,c,l),f=Th(i.shape,d,s,n,r,a,o),p=Mh(i,d,s.shape,n,r,a,o);if(null!=h){return[f,p,$h(b,d)]}return[f,p]},v={x:p,filter:f,bias:b,preluActivationWeights:y},k={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o,activation:l,leakyreluAlpha:h};if(null==i){return Ll((e,t,n)=>{let r=ta.runKernel(hr,v,k);return n([t,e,r]),g&&(r=ti(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:w}})(p,f)}return Ll((e,t,n,r)=>{let s=ta.runKernel(hr,v,k);return r([t,e,s,n]),g&&(s=ti(s,[s.shape[1],s.shape[2],s.shape[3]])),{value:s,gradFunc:w}})(p,f,b)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dh=pa({fusedMatMul_:function({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:s,activation:a="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(!1===_h(ta.state.gradientDepth,a)){let l=ai(e,t,n,r);return null!=s&&(l=Eo(l,s)),Ah(l,a,o,i)}let l=ha(e,"a","fused matMul"),c=ha(t,"b","fused matMul");[l,c]=js(l,c);const h=n?l.shape[l.rank-2]:l.shape[l.rank-1],f=r?c.shape[c.rank-1]:c.shape[c.rank-2],p=n?l.shape[l.rank-1]:l.shape[l.rank-2],g=r?c.shape[c.rank-2]:c.shape[c.rank-1],m=l.shape.slice(0,-2),b=c.shape.slice(0,-2),y=d(m),w=d(b);u(h===f,()=>`Error in fused matMul: inner shapes (${h}) and (${f}) of Tensors with shapes ${l.shape} and ${c.shape} and transposeA=${n} and transposeB=${r} must match.`);const v=Ki(l.shape.slice(0,-2),c.shape.slice(0,-2)).concat([p,g]),k=ti(l,n?[y,h,p]:[y,p,h]),x=ti(c,r?[w,g,f]:[w,f,g]);let E,S;null!=s&&(E=ha(s,"bias","fused matMul"),[E]=js(E,l),Ki(v,E.shape)),null!=o&&(S=ha(o,"prelu weights","fused matMul"));const $=(e,t)=>{const[o,i,l,u]=t,c=Sh(ti(e,l.shape),l,a);let h,d;if(n||r?!n&&r?(h=ai(c,i,!1,!1),d=ai(c,o,!0,!1)):n&&!r?(h=ai(i,c,!1,!0),d=ai(o,c,!1,!1)):(h=ai(i,c,!0,!0),d=ai(c,o,!0,!0)):(h=ai(c,i,!1,!0),d=ai(o,c,!0,!1)),null!=s){return[h,d,$h(u,c)]}return[h,d]},A={a:k,b:x,bias:E,preluActivationWeights:S},_={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:i};if(null==s){return Ll((e,t,n)=>{const r=ta.runKernel(ur,A,_);return n([e,t,r]),{value:ti(r,v),gradFunc:$}})(k,x)}return Ll((e,t,n,r)=>{const s=ta.runKernel(ur,A,_);return r([e,t,s,n]),{value:ti(s,v),gradFunc:$}})(k,x,E)}}),Rh=Object.freeze(Object.defineProperty({__proto__:null,conv2d:Ih,depthwiseConv2d:Nh,matMul:Dh},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bh=pa({hammingWindow_:
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e){return kh(e,.54,.46)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fh=pa({hannWindow_:function(e){return kh(e,.5,.5)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ch=pa({frame_:function(e,t,n,r=!1,s=0){let a=0;const o=[];for(;a+t<=e.size;)o.push(ii(e,a,t)),a+=n;if(r)for(;a<e.size;){const r=a+t-e.size,i=si([ii(e,a,t-r),vi([r],s)]);o.push(i),a+=n}return 0===o.length?Hc([],[0,t]):ti(si(o),[o.length,t])}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oh=pa({stft_:function(e,t,n,r,s=Fh){null==r&&(r=vh(t));const a=Ch(e,t,n),o=Ao(a,s(t));return Lc(o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ph=pa({cropAndResize_:function(e,t,n,r,s="bilinear",a=0){const o=ha(e,"image","cropAndResize"),i=ha(t,"boxes","cropAndResize","float32"),l=ha(n,"boxInd","cropAndResize","int32"),c=i.shape[0];u(4===o.rank,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),u(2===i.rank&&4===i.shape[1],()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${i.shape}.`),u(1===l.rank&&l.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${i.shape}.`),u(2===r.length,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),u(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),u("bilinear"===s||"nearest"===s,()=>`method must be bilinear or nearest, but was ${s}`);const h={image:o,boxes:i,boxInd:l},d={method:s,extrapolationValue:a,cropSize:r};return ta.runKernel(ze,h,d)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zh=pa({flipLeftRight_:function(e){const t=ha(e,"image","flipLeftRight","float32");u(4===t.rank,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return ta.runKernel(it,n,{})}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lh=pa({grayscaleToRGB_:function(e){const t=ha(e,"image","grayscaleToRGB"),n=t.rank-1,r=t.shape[n];u(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),u(1===r,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);const s=new Array(t.rank);return s.fill(1,0,n),s[n]=3,xl(t,s)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uh=pa({rgbToGrayscale_:function(e){const t=ha(e,"image","RGBToGrayscale"),n=t.rank-1,r=t.shape[n];u(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),u(3===r,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);const s=t.dtype,a=vo(t,"float32"),o=Vc([.2989,.587,.114]);let i;switch(t.rank){case 2:i=Yi("ij,j->i",a,o);break;case 3:i=Yi("ijk,k->ij",a,o);break;case 4:i=Yi("ijkl,l->ijk",a,o);break;case 5:i=Yi("ijklm,m->ijkl",a,o);break;case 6:i=Yi("ijklmn,n->ijklm",a,o);break;default:throw new Error("Not a valid tensor rank.")}return i=vl(i,-1),vo(i,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wh=pa({rotateWithOffset_:function(e,t,n=0,r=.5){const s=ha(e,"image","rotateWithOffset","float32");u(4===s.rank,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${s.rank}.`);const a={image:s},o={radians:t,fillValue:n,center:r};return ta.runKernel(lr,a,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gh(e,t,n,r,s,a){null==r&&(r=.5),null==s&&(s=Number.NEGATIVE_INFINITY),null==a&&(a=0);const o=e.shape[0];return n=Math.min(n,o),u(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),u(2===e.rank,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),u(4===e.shape[1],()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),u(1===t.rank,()=>"scores must be a 1D tensor"),u(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),u(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:s,softNmsSigma:a}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kh=pa({nonMaxSuppression_:function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const a=ha(e,"boxes","nonMaxSuppression","float32"),o=ha(t,"scores","nonMaxSuppression","float32"),i=Gh(a,o,n,r,s),l={maxOutputSize:n=i.maxOutputSize,iouThreshold:r=i.iouThreshold,scoreThreshold:s=i.scoreThreshold};return ta.runKernel(Ht,{boxes:a,scores:o},l)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qh(e,t,n){const r=function(e,t,n){return function(e,t,n){let r=0,s=e.length,a=0,o=!1;for(;r<s;){a=r+(s-r>>>1);const i=n(t,e[a]);i>0?r=a+1:(s=a,o=!i)}return o?r:-r-1}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e,t,n||jh)}(e,t,n),s=r<0?-(r+1):r;e.splice(s,0,t)}function jh(e,t){return e>t?1:e<t?-1:0}function Vh(e,t,n,r,s){return Zh(e,t,n,r,s,0)}function Hh(e,t,n,r,s,a){return Zh(e,t,n,r,s,0,!1,a,!0)}function Jh(e,t,n,r,s,a){return Zh(e,t,n,r,s,a,!0)}function Zh(e,t,n,r,s,a,o=!1,i=!1,l=!1){const u=[];for(let m=0;m<t.length;m++)t[m]>s&&u.push({score:t[m],boxIndex:m,suppressBeginIndex:0});u.sort(Qh);const c=a>0?-.5/a:0,h=[],d=[];for(;h.length<n&&u.length>0;){const t=u.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<s)break;let i=!1;for(let l=h.length-1;l>=o;--l){const n=Yh(e,a,h[l]);if(n>=r){i=!0;break}if(t.score=t.score*Xh(r,c,n),t.score<=s)break}t.suppressBeginIndex=h.length,i||(t.score===n?(h.push(a),d.push(t.score)):t.score>s&&qh(u,t,Qh))}const f=h.length,p=n-f;i&&p>0&&(h.push(...new Array(p).fill(0)),d.push(...new Array(p).fill(0)));const g={selectedIndices:h};return o&&(g.selectedScores=d),l&&(g.validOutputs=f),g}function Yh(e,t,n){const r=e.subarray(4*t,4*t+4),s=e.subarray(4*n,4*n+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),i=Math.max(r[0],r[2]),l=Math.max(r[1],r[3]),u=Math.min(s[0],s[2]),c=Math.min(s[1],s[3]),h=Math.max(s[0],s[2]),d=Math.max(s[1],s[3]),f=(i-a)*(l-o),p=(h-u)*(d-c);if(f<=0||p<=0)return 0;const g=Math.max(a,u),m=Math.max(o,c),b=Math.min(i,h),y=Math.min(l,d),w=Math.max(b-g,0)*Math.max(y-m,0);return w/(f+p-w)}function Xh(e,t,n){const r=Math.exp(t*n*n);return n<=e?r:0}function Qh(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ed=async function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const a=ha(e,"boxes","nonMaxSuppressionAsync"),o=ha(t,"scores","nonMaxSuppressionAsync"),i=Gh(a,o,n,r,s);n=i.maxOutputSize,r=i.iouThreshold,s=i.scoreThreshold;const l=await Promise.all([a.data(),o.data()]),u=l[0],c=l[1],{selectedIndices:h}=Vh(u,c,n,r,s);return a!==e&&a.dispose(),o!==t&&o.dispose(),Vc(h,"int32")};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const td=pa({nonMaxSuppressionWithScore_:function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=0){const o=ha(e,"boxes","nonMaxSuppression"),i=ha(t,"scores","nonMaxSuppression"),l=Gh(o,i,n,r,s,a),u={boxes:o,scores:i},c={maxOutputSize:n=l.maxOutputSize,iouThreshold:r=l.iouThreshold,scoreThreshold:s=l.scoreThreshold,softNmsSigma:a=l.softNmsSigma},h=ta.runKernel(Zt,u,c);return{selectedIndices:h[0],selectedScores:h[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nd=async function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=0){const o=ha(e,"boxes","nonMaxSuppressionAsync"),i=ha(t,"scores","nonMaxSuppressionAsync"),l=Gh(o,i,n,r,s,a);n=l.maxOutputSize,r=l.iouThreshold,s=l.scoreThreshold,a=l.softNmsSigma;const u=await Promise.all([o.data(),i.data()]),c=u[0],h=u[1],{selectedIndices:d,selectedScores:f}=Jh(c,h,n,r,s,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:Vc(d,"int32"),selectedScores:Vc(f)}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rd=pa({nonMaxSuppressionPadded_:function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=!1){const o=ha(e,"boxes","nonMaxSuppression"),i=ha(t,"scores","nonMaxSuppression"),l=Gh(o,i,n,r,s,null),u={boxes:o,scores:i},c={maxOutputSize:l.maxOutputSize,iouThreshold:l.iouThreshold,scoreThreshold:l.scoreThreshold,padToMaxOutputSize:a},h=ta.runKernel(Jt,u,c);return{selectedIndices:h[0],validOutputs:h[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sd=async function(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=!1){const o=ha(e,"boxes","nonMaxSuppressionAsync"),i=ha(t,"scores","nonMaxSuppressionAsync"),l=Gh(o,i,n,r,s,null),u=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,[d,f]=await Promise.all([o.data(),i.data()]),{selectedIndices:p,validOutputs:g}=Hh(d,f,u,c,h,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:Vc(p,"int32"),validOutputs:dl(g,"int32")}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ad=pa({resizeBilinear_:function(e,t,n=!1,r=!1){const s=ha(e,"images","resizeBilinear");u(3===s.rank||4===s.rank,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${s.rank}.`),u(2===t.length,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),u(!1===r||!1===n,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=s,o=!1;3===s.rank&&(o=!0,a=ti(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const i={images:a},l={alignCorners:n,halfPixelCenters:r,size:t},c=ta.runKernel(gn,i,l);return o?ti(c,[c.shape[1],c.shape[2],c.shape[3]]):c}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const od=pa({resizeNearestNeighbor_:function(e,t,n=!1,r=!1){const s=ha(e,"images","resizeNearestNeighbor");u(3===s.rank||4===s.rank,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${s.rank}.`),u(2===t.length,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),u("float32"===s.dtype||"int32"===s.dtype,()=>"`images` must have `int32` or `float32` as dtype"),u(!1===r||!1===n,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=s,o=!1;3===s.rank&&(o=!0,a=ti(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const i={images:a},l={alignCorners:n,halfPixelCenters:r,size:t},c=ta.runKernel(fn,i,l);return o?ti(c,[c.shape[1],c.shape[2],c.shape[3]]):c}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const id=pa({threshold_:function(e,t="binary",n=!1,r=.5){const s=ha(e,"image","threshold"),a=s.shape[0]*s.shape[1];let o,i,l,c,h=Ao(Vc([r]),255);if(u(3===s.rank,()=>`Error in threshold: image must be rank 3,but got rank ${s.rank}.`),u(3===s.shape[2]||1===s.shape[2],()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${s.shape[2]}.`),u("int32"===s.dtype||"float32"===s.dtype,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${s.dtype}.`),u("otsu"===t||"binary"===t,()=>`Method must be binary or otsu, but was ${t}`),3===s.shape[2]){[o,i,l]=zc(s,[1,1,1],-1);const e=Ao(o,.2989),t=Ao(i,.587),n=Ao(l,.114);c=Eo(Eo(e,t),n)}else c=e;if("otsu"===t){h=function(e,t){let n,r,s,a,o,i,l=Vc([-1]),u=Vc([0]),c=Vc([0]);for(let h=0;h<e.size-1;h++){n=ii(e,0,h+1),r=ii(e,h+1),o=$o(gl(n),t),i=$o(gl(r),t);const d=gl(Ao(n,fc(0,n.size)));s=$o(d,gl(n));const f=vi(r.shape,n.size),p=Eo(fc(0,r.size),f),g=Ao(r,p);a=$o(gl(g),gl(r));const m=ql(s,a),b=ql(s,a),y=Ao(o,i);c=Ao(Ao(y,m),b);const w=Al(c,u);u=Vi(w,c,u),l=Vi(w,Vc([h]),l)}return l}(gi(vo(Ec(c),"int32"),ba([]),256),a)}const d=n?Bl(c,h):Al(c,h);return vo(Ao(d,255),"int32")}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ld=pa({transform_:function(e,t,n="nearest",r="constant",s=0,a){const o=ha(e,"image","transform","float32"),i=ha(t,"transforms","transform","float32");u(4===o.rank,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),u(2===i.rank&&(i.shape[0]===o.shape[0]||1===i.shape[0])&&8===i.shape[1],()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),u(null==a||2===a.length,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const l={image:o,transforms:i},c={interpolation:n,fillMode:r,fillValue:s,outputShape:a};return ta.runKernel(er,l,c)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ud=pa({bandPart_:function(e,t,n){const r=ha(e,"a","bandPart");u(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);const s=r.shape,[a,o]=r.shape.slice(-2);let i,l;"number"==typeof t?(u(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),u(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),i=ha(t<0?a:t,"numLower","bandPart")):(u("int32"===t.dtype,()=>"bandPart(): numLower's dtype must be an int32."),i=Vi(Rl(t,0),a,uu(t,a))),"number"==typeof n?(u(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),u(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),l=ha(n<0?o:n,"numUpper","bandPart")):(u("int32"===n.dtype,()=>"bandPart(): numUpper's dtype must be an int32."),l=Vi(Rl(n,0),o,uu(n,o)));const c=ti(fc(0,a,1,"int32"),[-1,1]),h=fc(0,o,1,"int32"),d=ql(c,h),f=Hl(Bl(d,i),_l(d,Wl(l))),p=ou([a,o],r.dtype);return ti(Gc(lh(ti(r,[-1,a,o])).map(e=>Vi(f,e,p))),s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cd=pa({gramSchmidt_:function(e){let t;if(Array.isArray(e)){t=!1,u(null!=e&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const n=e[0].shape[0];for(let t=1;t<e.length;++t)u(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=zc(e,e.shape[0],0).map(e=>Wc(e,[0]));u(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],r=e;for(let s=0;s<e.length;++s)n.push(ta.tidy(()=>{let e=r[s];if(s>0)for(let t=0;t<s;++t){const r=Ao(gl(Ao(n[t],e)),n[t]);e=ql(e,r)}return $o(e,bl(e,"euclidean"))}));return t?Gc(n,0):n}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hd(e,t=!1){return ta.tidy(()=>{u(2===e.shape.length,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],r=e.shape[1];let s=El(n),a=ko(e);const o=Hc([[1]],[1,1]);let i=ko(o);const l=n>=r?r:n;for(let e=0;e<l;++e){const t=a,l=i,u=s;[i,a,s]=ta.tidy(()=>{const t=ii(a,[e,e],[n-e,1]),l=bl(t),u=ii(a,[e,e],[1,1]),c=Vi(Al(u,0),Hc([[-1]]),Hc([[1]])),h=ql(u,Ao(c,l)),d=$o(t,h);i=1===d.shape[0]?ko(o):si([o,ii(d,[1,0],[d.shape[0]-1,d.shape[1]])],0);const f=Wl($o(ai(c,h),l)),p=ii(a,[e,0],[n-e,r]),g=Ao(f,i),m=ph(i);if(0===e)a=ql(p,ai(g,ai(m,p)));else{const t=ql(p,ai(g,ai(m,p)));a=si([ii(a,[0,0],[e,r]),t],0)}const b=ph(g),y=ii(s,[0,e],[n,s.shape[1]-e]);if(0===e)s=ql(y,ai(ai(y,i),b));else{const t=ql(y,ai(ai(y,i),b));s=si([ii(s,[0,0],[n,e]),t],1)}return[i,a,s]}),xa([t,l,u])}return!t&&n>r&&(s=ii(s,[0,0],[n,r]),a=ii(a,[0,0],[r,r])),[s,a]})}const dd=pa({qr_:function(e,t=!1){if(u(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),2===e.rank)return hd(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),r=lh(ti(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),s=[],a=[];r.forEach(e=>{const[n,r]=hd(e,t);s.push(n),a.push(r)});return[ti(Gc(s,0),e.shape),ti(Gc(a,0),e.shape)]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var fd,pd;(pd=fd||(fd={}))[pd.NONE=0]="NONE",pd[pd.MEAN=1]="MEAN",pd[pd.SUM=2]="SUM",pd[pd.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS";const gd=pa({computeWeightedLoss_:function(e,t,n=fd.SUM_BY_NONZERO_WEIGHTS){const r=ha(e,"losses","computeWeightedLoss");let s=null;null!=t&&(s=ha(t,"weights","computeWeightedLoss"));const a=null==s?r:Ao(r,s);if(n===fd.NONE)return a;if(n===fd.SUM)return gl(a);if(n===fd.MEAN){if(null==s)return au(a);{const e=r.size/s.size,t=$o(gl(a),gl(s));return e>1?$o(t,dl(e)):t}}if(n===fd.SUM_BY_NONZERO_WEIGHTS){if(null==s)return $o(gl(a),dl(r.size));{const e=Ao(s,iu(r.shape)),t=vo(gl(gu(e,dl(0))),"float32");return $o(gl(a),t)}}throw Error(`Unknown reduction: ${n}`)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const md=pa({absoluteDifference_:function(e,t,n,r=fd.SUM_BY_NONZERO_WEIGHTS){const s=ha(e,"labels","absoluteDifference"),a=ha(t,"predictions","absoluteDifference");let o=null;null!=n&&(o=ha(n,"weights","absoluteDifference")),c(s.shape,a.shape,"Error in absoluteDifference: ");const i=_o(ql(s,a));return gd(i,o,r)}});const bd=pa({cosineDistance_:function(e,t,n,r,s=fd.SUM_BY_NONZERO_WEIGHTS){const a=ha(e,"labels","cosineDistance"),o=ha(t,"predictions","cosineDistance");let i=null;null!=r&&(i=ha(r,"weights","cosineDistance")),c(a.shape,o.shape,"Error in cosineDistance: ");const l=dl(1),u=ql(l,gl(Ao(a,o),n,!0));return gd(u,i,s)}});const yd=pa({hingeLoss_:function(e,t,n,r=fd.SUM_BY_NONZERO_WEIGHTS){let s=ha(e,"labels","hingeLoss");const a=ha(t,"predictions","hingeLoss");let o=null;null!=n&&(o=ha(n,"weights","hingeLoss")),c(s.shape,a.shape,"Error in hingeLoss: ");const i=dl(1);s=ql(Ao(dl(2),s),i);const l=mc(ql(i,Ao(s,a)));return gd(l,o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wd=pa({huberLoss_:function(e,t,n,r=1,s=fd.SUM_BY_NONZERO_WEIGHTS){const a=ha(e,"labels","huberLoss"),o=ha(t,"predictions","huberLoss");let i=null;null!=n&&(i=ha(n,"weights","huberLoss")),c(a.shape,o.shape,"Error in huberLoss: ");const l=dl(r),u=_o(ql(o,a)),h=uu(u,l),d=ql(u,h),f=Eo(Ao(dl(.5),pl(h)),Ao(l,d));return gd(f,i,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vd=pa({logLoss_:function(e,t,n,r=1e-7,s=fd.SUM_BY_NONZERO_WEIGHTS){const a=ha(e,"labels","logLoss"),o=ha(t,"predictions","logLoss");let i=null;null!=n&&(i=ha(n,"weights","logLoss")),c(a.shape,o.shape,"Error in logLoss: ");const l=dl(1),u=dl(r),h=Wl(Ao(a,Ol(Eo(o,u)))),d=Ao(ql(l,a),Ol(Eo(ql(l,o),u))),f=ql(h,d);return gd(f,i,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kd=pa({meanSquaredError_:function(e,t,n,r=fd.SUM_BY_NONZERO_WEIGHTS){const s=ha(e,"labels","meanSquaredError"),a=ha(t,"predictions","meanSquaredError");let o=null;null!=n&&(o=ha(n,"weights","meanSquaredError")),c(s.shape,a.shape,"Error in meanSquaredError: ");const i=Uc(s,a);return gd(i,o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xd=pa({sigmoidCrossEntropy_:function(e,t,n,r=0,s=fd.SUM_BY_NONZERO_WEIGHTS){let a=ha(e,"multiClassLabels","sigmoidCrossEntropy");const o=ha(t,"logits","sigmoidCrossEntropy");let i=null;if(null!=n&&(i=ha(n,"weights","sigmoidCrossEntropy")),c(a.shape,o.shape,"Error in sigmoidCrossEntropy: "),r>0){const e=dl(r),t=dl(1),n=dl(.5);a=Eo(Ao(a,ql(t,e)),Ao(n,e))}const l=function(e,t){const n=ha(e,"labels","sigmoidCrossEntropyWithLogits"),r=ha(t,"logits","sigmoidCrossEntropyWithLogits");c(n.shape,r.shape,"Error in sigmoidCrossEntropyWithLogits: ");const s=mc(r),a=Ao(r,n),o=Pl(wl(Wl(_o(r))));return Eo(ql(s,a),o)}(a,o);return gd(l,i,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ed=pa({softmaxCrossEntropy_:function(e,t,n,r=0,s=fd.SUM_BY_NONZERO_WEIGHTS){let a=ha(e,"onehotLabels","softmaxCrossEntropy");const o=ha(t,"logits","softmaxCrossEntropy");let i=null;if(null!=n&&(i=ha(n,"weights","softmaxCrossEntropy")),c(a.shape,o.shape,"Error in softmaxCrossEntropy: "),r>0){const e=dl(r),t=dl(1),n=dl(a.shape[1]);a=Eo(Ao(a,ql(t,e)),$o(e,n))}const l=function(e,t,n=-1){if(-1===n&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return Ll((e,t,r)=>{const s=Vl(t,[n],!0),a=ql(vo(t,"float32"),s);r([e,a]);const o=Wl(Ao(a,e));return{value:gl(o,[n]),gradFunc:(e,t)=>{const[r,s]=t,a=sl(e.shape,[n]);return[Ao(ti(e,a),ql(vo(r,"float32"),wl(s))),Ao(ti(e,a),ql(wl(s),vo(r,"float32")))]}}})(e,t)}(a,o);return gd(l,i,s)}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sd={fft:Cc,ifft:Oc,rfft:Lc,irfft:Pc},$d={hammingWindow:Bh,hannWindow:Fh,frame:Ch,stft:Oh},Ad={flipLeftRight:zh,grayscaleToRGB:Lh,resizeNearestNeighbor:od,resizeBilinear:ad,rgbToGrayscale:Uh,rotateWithOffset:Wh,cropAndResize:Ph,nonMaxSuppression:Kh,nonMaxSuppressionAsync:ed,nonMaxSuppressionWithScore:td,nonMaxSuppressionWithScoreAsync:nd,nonMaxSuppressionPadded:rd,nonMaxSuppressionPaddedAsync:sd,threshold:id,transform:ld},_d={bandPart:ud,gramSchmidt:cd,qr:dd},Id={absoluteDifference:md,computeWeightedLoss:gd,cosineDistance:bd,hingeLoss:yd,huberLoss:wd,logLoss:vd,meanSquaredError:kd,sigmoidCrossEntropy:xd,softmaxCrossEntropy:Ed},Md={sparseFillEmptyRows:pa({sparseFillEmptyRows_:function(e,t,n,r){const s=ha(e,"indices","sparseFillEmptyRows","int32"),a=ha(t,"values","sparseFillEmptyRows"),o=ha(n,"denseShape","sparseFillEmptyRows","int32"),i=ha(r,"defaultValue","sparseFillEmptyRows",a.dtype);if(2!==s.rank)throw new Error(`Indices should be Tensor2D but received shape\n        ${s.shape}`);if(1!==a.rank)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(1!==o.rank)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(0!==i.rank)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const l={indices:s,values:a,denseShape:o,defaultValue:i},u=ta.runKernel(On,l);return{outputIndices:u[0],outputValues:u[1],emptyRowIndicator:u[2],reverseIndexMap:u[3]}}}),sparseReshape:pa({sparseReshape_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ha(e,"inputIndices","sparseReshape","int32"),s=ha(t,"inputShape","sparseReshape","int32"),a=ha(n,"newShape","sparseReshape","int32");if(2!==r.rank)throw new Error(`Input indices should be Tensor2D but received shape\n        ${r.shape}`);if(1!==s.rank)throw new Error(`Input shape should be Tensor1D but received shape ${s.shape}`);if(1!==a.rank)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const o={inputIndices:r,inputShape:s,newShape:a},i=ta.runKernel(Pn,o);return{outputIndices:i[0],outputShape:i[1]}}}),sparseSegmentMean:pa({sparseSegmentMean_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ha(e,"data","sparseSegmentMean"),s=ha(t,"indices","sparseSegmentMean","int32"),a=ha(n,"segmentIds","sparseSegmentMean","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(1!==s.rank)throw new Error(`Indices should be Tensor1D but received shape\n          ${s.shape}`);if(1!==a.rank)throw new Error(`Segment ids should be Tensor1D but received shape\n          ${a.shape}`);const o={data:r,indices:s,segmentIds:a};return ta.runKernel(zn,o)}}),sparseSegmentSum:pa({sparseSegmentSum_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ha(e,"data","sparseSegmentSum"),s=ha(t,"indices","sparseSegmentSum","int32"),a=ha(n,"segmentIds","sparseSegmentSum","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(1!==s.rank)throw new Error(`Indices should be Tensor1D but received shape\n         ${s.shape}`);if(1!==a.rank)throw new Error(`Segment ids should be Tensor1D but received shape\n         ${a.shape}`);const o={data:r,indices:s,segmentIds:a};return ta.runKernel(Ln,o)}})},Td={stringNGrams:pa({stringNGrams_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r,s,a,o,i){const l=ha(e,"data","stringNGrams","string");if("string"!==l.dtype)throw new Error("Data must be of datatype string");if(1!==l.shape.length)throw new Error(`Data must be a vector, saw: ${l.shape}`);const u=ha(t,"dataSplits","stringNGrams");if("int32"!==u.dtype)throw new Error("Data splits must be of datatype int32");const c={separator:n,nGramWidths:r,leftPad:s,rightPad:a,padWidth:o,preserveShortSequences:i},h={data:l,dataSplits:u},d=ta.runKernel(jn,h,c);return{nGrams:d[0],nGramsSplits:d[1]}}}),stringSplit:pa({stringSplit_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n=!0){const r=ha(e,"input","stringSplit","string"),s=ha(t,"delimiter","stringSplit","string");if(1!==r.rank)throw new Error(`Input should be Tensor1D but received shape ${r.shape}`);if(0!==s.rank)throw new Error(`Delimiter should be a scalar but received shape ${s.shape}`);const a={skipEmpty:n},o={input:r,delimiter:s},i=ta.runKernel(Vn,o,a);return{indices:i[0],values:i[1],shape:i[2]}}}),stringToHashBucketFast:pa({stringToHashBucketFast_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t){const n=ha(e,"input","stringToHashBucketFast","string"),r={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const s={input:n};return ta.runKernel(Hn,s,r)}}),staticRegexReplace:pa({staticRegexReplace_:
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r=!0){const s=ha(e,"input","staticRegexReplace","string"),a={pattern:t,rewrite:n,replaceGlobal:r};return ta.runKernel(Kn,{x:s},a)}})},Nd=new Map,Dd=new Map;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Rd{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class Bd{constructor(){this.classNameMap={}}static getMap(){return null==Bd.instance&&(Bd.instance=new Bd),Bd.instance}static register(e){Bd.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function Fd(e,t,n){u(null!=e.className,()=>"Class being registered does not have the static className property defined."),u("string"==typeof e.className,()=>"className is required to be a string, but got type "+typeof e.className),u(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),void 0===t&&(t="Custom"),void 0===n&&(n=e.className);const r=t+">"+n;return Bd.register(e),Nd.set(r,e),Dd.set(e,r),e}const Cd=Object.freeze(Object.defineProperty({__proto__:null,Serializable:Rd,SerializationMap:Bd,getRegisteredName:function(e){return Dd.has(e)?Dd.get(e):e.className},registerClass:Fd},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Od extends Rd{minimize(e,t=!1,n){const{value:r,grads:s}=this.computeGradients(e,n);if(null!=n){const e=n.map(e=>({name:e.name,tensor:s[e.name]}));this.applyGradients(e)}else this.applyGradients(s);return xa(s),t?r:(r.dispose(),null)}get iterations(){return null==this.iterations_&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return zl(e,t)}dispose(){null!=this.iterations_&&xa(this.iterations_)}async saveIterations(){return null==this.iterations_&&(this.iterations_=0),{name:"iter",tensor:dl(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(Od,Symbol.hasInstance,{value:e=>null!=e.minimize&&null!=e.computeGradients&&null!=e.applyGradients});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Pd extends Od{static get className(){return"Adadelta"}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],null==n&&(this.epsilon=ta.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=ta.registeredVariables[t],s=!1;null==this.accumulatedGrads[n]&&(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:ka(()=>Hi(r).variable(s))}),null==this.accumulatedUpdates[n]&&(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:ka(()=>Hi(r).variable(s))});const a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;const o=this.accumulatedGrads[n].variable,i=this.accumulatedUpdates[n].variable;ka(()=>{const e=Eo(Ao(o,this.rho),Ao(pl(a),1-this.rho)),t=Ao($o(fl(Eo(i,this.epsilon)),fl(Eo(o,this.epsilon))),a),n=Eo(Ao(i,this.rho),Ao(pl(t),1-this.rho));o.assign(e),i.assign(n);const s=Eo(Ao(t,-this.learningRate),r);r.assign(s)})}),this.incrementIterations()}dispose(){null!=this.accumulatedUpdates&&(xa(this.accumulatedGrads.map(e=>e.variable)),xa(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){const t=(e=await this.extractIterations(e)).length/2,n=!1;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedUpdates=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class zd extends Od{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=ta.registeredVariables[t];if(null==this.accumulatedGrads[n]){const e=!1;this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:ka(()=>vi(r.shape,this.initialAccumulatorValue).variable(e))}}const s=Array.isArray(e)?e[n].tensor:e[t];if(null==s)return;const a=this.accumulatedGrads[n].variable;ka(()=>{const e=Eo(a,pl(s));a.assign(e);const t=Eo(Ao($o(s,fl(Eo(e,ta.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){null!=this.accumulatedGrads&&xa(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(false)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ld extends Od{static get className(){return"Adam"}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],ka(()=>{this.accBeta1=dl(t).variable(),this.accBeta2=dl(n).variable()}),null==r&&(this.epsilon=ta.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);ka(()=>{const n=ql(1,this.accBeta1),r=ql(1,this.accBeta2);t.forEach((t,s)=>{const a=ta.registeredVariables[t],o=!1;null==this.accumulatedFirstMoment[s]&&(this.accumulatedFirstMoment[s]={originalName:`${t}/m`,variable:ka(()=>Hi(a).variable(o))}),null==this.accumulatedSecondMoment[s]&&(this.accumulatedSecondMoment[s]={originalName:`${t}/v`,variable:ka(()=>Hi(a).variable(o))});const i=Array.isArray(e)?e[s].tensor:e[t];if(null==i)return;const l=this.accumulatedFirstMoment[s].variable,u=this.accumulatedSecondMoment[s].variable,c=Eo(Ao(l,this.beta1),Ao(i,1-this.beta1)),h=Eo(Ao(u,this.beta2),Ao(pl(i),1-this.beta2)),d=$o(c,n),f=$o(h,r);l.assign(c),u.assign(h);const p=Eo(Ao($o(d,Eo(fl(f),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(Ao(this.accBeta1,this.beta1)),this.accBeta2.assign(Ao(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),null!=this.accumulatedFirstMoment&&xa(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedSecondMoment&&xa(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),ka(()=>{this.accBeta1.assign(hl(this.beta1,this.iterations_+1)),this.accBeta2.assign(hl(this.beta2,this.iterations_+1))});const t=e.length/2,n=!1;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedSecondMoment=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ud extends Od{static get className(){return"Adamax"}constructor(e,t,n,r=null,s=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=s,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],ka(()=>{this.iteration=dl(0).variable(),this.accBeta1=dl(t).variable()}),null==r&&(this.epsilon=ta.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);ka(()=>{const n=ql(1,this.accBeta1),r=$o(-this.learningRate,Eo(Ao(this.iteration,this.decay),1));t.forEach((t,s)=>{const a=ta.registeredVariables[t],o=!1;null==this.accumulatedFirstMoment[s]&&(this.accumulatedFirstMoment[s]={originalName:`${t}/m`,variable:Hi(a).variable(o)}),null==this.accumulatedWeightedInfNorm[s]&&(this.accumulatedWeightedInfNorm[s]={originalName:`${t}/v`,variable:Hi(a).variable(o)});const i=Array.isArray(e)?e[s].tensor:e[t];if(null==i)return;const l=this.accumulatedFirstMoment[s].variable,u=this.accumulatedWeightedInfNorm[s].variable,c=Eo(Ao(l,this.beta1),Ao(i,1-this.beta1)),h=Ao(u,this.beta2),d=_o(i),f=su(h,d);l.assign(c),u.assign(f);const p=Eo(Ao($o(r,n),$o(c,Eo(f,this.epsilon))),a);a.assign(p)}),this.iteration.assign(Eo(this.iteration,1)),this.accBeta1.assign(Ao(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),null!=this.accumulatedFirstMoment&&xa(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedWeightedInfNorm&&xa(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Wd extends Od{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=Array.isArray(e)?e[n].tensor:e[t];if(null==r)return;const s=ta.registeredVariables[t];ka(()=>{const e=Eo(Ao(this.c,r),s);s.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,null!=this.c&&this.c.dispose(),this.c=Ea(dl(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(0!==(e=await this.extractIterations(e)).length)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Gd extends Wd{static get className(){return"Momentum"}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=dl(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=ta.registeredVariables[t];if(null==this.accumulations[n]){const e=!1;this.accumulations[n]={originalName:`${t}/momentum`,variable:ka(()=>Hi(r).variable(e))}}const s=this.accumulations[n].variable,a=Array.isArray(e)?e[n].tensor:e[t];null!=a&&ka(()=>{let e;const t=Eo(Ao(this.m,s),a);e=this.useNesterov?Eo(Ao(this.c,Eo(a,Ao(t,this.m))),r):Eo(Ao(this.c,t),r),s.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),null!=this.accumulations&&xa(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(false)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Kd extends Od{static get className(){return"RMSProp"}constructor(e,t=.9,n=0,r=null,s=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=s,null==r&&(this.epsilon=ta.backend.epsilon()),null==e)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=ta.registeredVariables[t],s=!1;null==this.accumulatedMeanSquares[n]&&(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:ka(()=>Hi(r).variable(s))}),null==this.accumulatedMoments[n]&&(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:ka(()=>Hi(r).variable(s))}),null==this.accumulatedMeanGrads[n]&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:ka(()=>Hi(r).variable(s))});const a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;const o=this.accumulatedMeanSquares[n].variable,i=this.accumulatedMoments[n].variable;ka(()=>{const e=Eo(Ao(o,this.decay),Ao(pl(a),1-this.decay));if(this.centered){const t=this.accumulatedMeanGrads[n].variable,s=Eo(Ao(t,this.decay),Ao(a,1-this.decay)),l=$o(Ao(a,this.learningRate),fl(ql(e,Eo(pl(s),this.epsilon)))),u=Eo(Ao(i,this.momentum),l);o.assign(e),t.assign(s),i.assign(u);const c=ql(r,u);r.assign(c)}else{const e=Eo(Ao(o,this.decay),Ao(pl(a),1-this.decay)),t=Eo(Ao(i,this.momentum),$o(Ao(a,this.learningRate),fl(Eo(e,this.epsilon))));o.assign(e),i.assign(t);const n=ql(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){null!=this.accumulatedMeanSquares&&xa(this.accumulatedMeanSquares.map(e=>e.variable)),null!=this.accumulatedMeanGrads&&this.centered&&xa(this.accumulatedMeanGrads.map(e=>e.variable)),null!=this.accumulatedMoments&&xa(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,n=!1;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedMoments=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.centered&&(this.accumulatedMeanGrads=e.slice(2*t,3*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}
/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qd=[Pd,zd,Ld,Ud,Gd,Kd,Wd];function jd(e){return new Promise(e=>setTimeout(e)).then(e)}class Vd{constructor(e){if(!J().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(Vd.URL_SCHEME)&&(e=e.slice(Vd.URL_SCHEME.length)),null!=e&&0!==e.length||(e="model"),this.modelJsonFileName=e+".json",this.weightDataFileName=e+".weights.bin"}async save(e){if("undefined"==typeof document)throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=wa.join(e.weightData),n=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const t=Oa(e,[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}]),r=window.URL.createObjectURL(new Blob([JSON.stringify(t)],{type:"application/json"})),s=null==this.modelJsonAnchor?document.createElement("a"):this.modelJsonAnchor;if(s.download=this.modelJsonFileName,s.href=r,await jd(()=>s.dispatchEvent(new MouseEvent("click"))),null!=e.weightData){const e=null==this.weightDataAnchor?document.createElement("a"):this.weightDataAnchor;e.download=this.weightDataFileName,e.href=n,await jd(()=>e.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:La(e)}}}}Vd.URL_SCHEME="downloads://";class Hd{constructor(e){if(null==e||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const n=new FileReader;n.onload=n=>{const r=JSON.parse(n.target.result),s=r.modelTopology;if(null==s)return void t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));if(null==r.weightsManifest)return void t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));if(0===this.weightsFiles.length)return void e({modelTopology:s});const a=za(r,e=>this.loadWeights(e));e(a)},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){const t=[],n=[];for(const a of e)t.push(...a.weights),n.push(...a.paths);const r=this.checkManifestAndWeightFiles(e),s=n.map(e=>this.loadWeightsFile(e,r[e]));return Promise.all(s).then(e=>[t,e])}loadWeightsFile(e,t){return new Promise((n,r)=>{const s=new FileReader;s.onload=e=>{const t=e.target.result;n(t)},s.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),s.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],n=this.weightsFiles.map(e=>Ca(e.name)),r={};for(const s of e)s.paths.forEach(e=>{const s=Ca(e);if(-1!==t.indexOf(s))throw new Error(`Duplicate file basename found in weights manifest: '${s}'`);if(t.push(s),-1===n.indexOf(s))throw new Error(`Weight file with basename '${s}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(s)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Jd(e,t,n,r){var s,a,o;u(null!=(s=e)&&Array.isArray(s)&&s.length>0,()=>"promises must be a none empty array"),o=r=null==r?1:r,u((a=n=null==n?0:n)>=0&&a<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${a}`),u(o>=0&&o<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${o}`),u(o>=a,()=>`startFraction must be no more than endFraction, but got startFraction ${a} and endFraction ${o}`);let i=0;return Promise.all(e.map(s=>(s.then(s=>{const a=n+ ++i/e.length*(r-n);return t(a),s}),s)))}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Zd(e,t){null==t&&(t={});const n=null==t.fetchFunc?J().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),s=(null==t.onProgress?await Promise.all(r):await Jd(r,t.onProgress,0,.5)).map(e=>e.arrayBuffer());return null==t.onProgress?await Promise.all(s):await Jd(s,t.onProgress,.5,1)}function Yd(e){return async(t,n="",r)=>{const s=t.map(()=>!1),a={},o=null!=r?r.map(()=>!1):[],i=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{const l="quantization"in e?e.quantization.dtype:e.dtype,u=ya[l]*d(e.shape),c=()=>{s[t]=!0,null==a[t]&&(a[t]=[]),a[t].push({manifestEntry:e,groupOffset:n,sizeBytes:u})};null!=r?r.forEach((t,n)=>{t===e.name&&(c(),o[n]=!0)}):c(),i.push(e.name),n+=u})}),!o.every(e=>e)){const e=r.filter((e,t)=>!o[t]);throw new Error(`Could not find weights in manifest with names: ${e.join(", ")}. \nManifest JSON has weights with names: ${i.join(", ")}.`)}const l=s.reduce((e,t,n)=>(t&&e.push(n),e),[]),u=[];l.forEach(e=>{t[e].paths.forEach(e=>{const t=n+(n.endsWith("/")?"":"/")+e;u.push(t)})});const c=await e(u),h={};let f=0;return l.forEach(e=>{const n=t[e].paths.length,r=new wa(c.slice(f,f+n));a[e].forEach(e=>{const t=_a(r.slice(e.groupOffset,e.groupOffset+e.sizeBytes),[e.manifestEntry]);for(const n in t)h[n]=t[n]}),f+=n}),h}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Wa.registerSaveRouter(e=>J().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Vd.URL_SCHEME)?function(e="model"){return new Vd(e)}(e.slice(Vd.URL_SCHEME.length)):null);class Xd{constructor(e,t){if(this.DEFAULT_METHOD="POST",null==t&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,null!=t.fetchFunc?(u("function"==typeof t.fetchFunc,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=J().platform.fetch,u(null!=e&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&u(2===e.length,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,null!=t.requestInit&&null!=t.requestInit.body)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;const n=Oa(e,[{paths:["./model.weights.bin"],weights:e.weightSpecs}]);if(t.body.append("model.json",new Blob([JSON.stringify(n)],{type:"application/json"}),"model.json"),null!=e.weightData){const n=wa.join(e.weightData);t.body.append("model.weights.bin",new Blob([n],{type:"application/octet-stream"}),"model.weights.bin")}const r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:La(e),responses:[r]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch(Lp){let t=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?t+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":t+=" Please make sure the server is serving valid JSON for this request.",new Error(t)}const n=t.modelTopology,r=t.weightsManifest;if(null==n&&null==r)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){if(this.loadOptions.streamWeights)return this.loadStream();return za(await this.loadModelJSON(),e=>this.loadWeights(e))}async loadStream(){const e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),n=Ua(e.weightsManifest);return Object.assign(Object.assign({},e),{weightSpecs:n,getWeightStream:()=>function(e,t){var n;const r=null==t.fetchFunc?J().platform.fetch:t.fetchFunc;let s,a=0;return null===(n=t.onProgress)||void 0===n||n.call(t,0),new ReadableStream({pull:async n=>{for(var o;a<e.length;){if(!s){const n=(await r(e[a],t.requestInit,{isBinary:!0})).body;s=n.getReader()}const{done:i,value:l}=await s.read();if(!i)return void n.enqueue(l);a++,s=void 0,null===(o=t.onProgress)||void 0===o||o.call(t,a/e.length)}n.close()}})}(t,this.loadOptions)})}async getWeightUrls(e){const t=Array.isArray(this.path)?this.path[1]:this.path,[n,r]=function(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),r=e.substring(0,t),s=n>t?e.substring(n):"";return[r+"/",s]}(t),s=this.weightPathPrefix||n,a=[],o=[];for(const i of e)for(const e of i.paths)null!=this.weightUrlConverter?o.push(this.weightUrlConverter(e)):a.push(s+e+r);return this.weightUrlConverter&&a.push(...await Promise.all(o)),a}async loadWeights(e){const t=await this.getWeightUrls(e);return[Ua(e),await Zd(t,this.loadOptions)]}}function Qd(e){return null!=e.match(Xd.URL_SCHEME_REGEX)}Xd.URL_SCHEME_REGEX=/^https?:\/\//;const ef=(e,t)=>{if("undefined"==typeof fetch&&(null==t||null==t.fetchFunc))return null;{let n=!0;if(n=Array.isArray(e)?e.every(e=>Qd(e)):Qd(e),n)return tf(e,t)}return null};function tf(e,t){return new Xd(e,t)}Wa.registerSaveRouter(ef),Wa.registerLoadRouter(ef);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class nf{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class rf{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class sf{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function af(e,t,n,r){if(1===arguments.length){return null!=e.modelTopology||null!=e.weightSpecs?new nf(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new nf({modelTopology:e}))}return console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new nf({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r})}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const of=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:wa,browserFiles:function(e){return new Hd(e)},browserHTTPRequest:function(e,t){return tf(e,t)},concatenateArrayBuffers:function(e){return wa.join(e)},copyModel:async function(e,t){return po(e,t,!1)},decodeWeights:_a,decodeWeightsStream:Da,encodeWeights:async function(e,t){const n=[],r=[],s=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let a=0;a<s.length;++a){const o=s[a],i=Array.isArray(e)?e[a].tensor:e[o];if("float32"!==i.dtype&&"int32"!==i.dtype&&"bool"!==i.dtype&&"string"!==i.dtype&&"complex64"!==i.dtype)throw new Error(`Unsupported dtype in weight '${o}': ${i.dtype}`);const l={name:o,shape:i.shape,dtype:i.dtype};if("string"===i.dtype){const e=new Promise(async e=>{const t=await i.bytes(),n=t.reduce((e,t)=>e+t.length,0)+4*t.length,r=new Uint8Array(n);let s=0;for(let a=0;a<t.length;a++){const e=t[a],n=new Uint8Array(new Uint32Array([e.length]).buffer);r.set(n,s),s+=4,r.set(e,s),s+=e.length}e(r)});r.push(e)}else r.push(i.data());null!=t&&(l.group=t),n.push(l)}return{data:Ra(await Promise.all(r)),specs:n}},fromMemory:function(e,t,n,r){return new sf(af(...arguments))},fromMemorySync:af,getLoadHandlers:(e,t)=>Wa.getLoadHandlers(e,t),getModelArtifactsForJSON:za,getModelArtifactsForJSONSync:Pa,getModelArtifactsInfoForJSON:La,getSaveHandlers:e=>Wa.getSaveHandlers(e),getWeightSpecs:Ua,http:tf,isHTTPScheme:Qd,listModels:async function(){const e=ho.getSchemes(),t={};for(const n of e){const e=await ho.getManager(n).listModels();for(const r in e){t[n+co+r]=e[r]}}return t},loadWeights:async function(e,t="",n,r){return Yd(e=>Zd(e,{requestInit:r}))(e,t,n)},moveModel:async function(e,t){return po(e,t,!0)},registerLoadRouter:e=>Wa.registerLoadRouter(e),registerSaveRouter:e=>Wa.registerSaveRouter(e),removeModel:async function(e){const t=fo(e);return ho.getManager(t.scheme).removeModel(t.path)},weightsLoaderFactory:Yd,withSaveHandler:function(e){return new rf(e)},withSaveHandlerSync:function(e){return new rf(e)}},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lf=pa({confusionMatrix_:function(e,t,n){const r=ha(e,"labels","confusionMatrix"),s=ha(t,"predictions","confusionMatrix");u(null==n||n>0&&Number.isInteger(n),()=>`If provided, numClasses must be a positive integer, but got ${n}`),u(1===r.rank,()=>`Expected the rank of labels to be 1, but got ${r.rank}`),u(1===s.rank,()=>`Expected the rank of predictions to be 1, but got ${s.rank}`),u(r.shape[0]===s.shape[0],()=>`Mismatch in the number of examples: ${r.shape[0]} vs. ${s.shape[0]}. Labels and predictions should have the same number of elements.`),u(n>0&&Number.isInteger(n),()=>`numClasses is required to be a positive integer, but got ${n}`);const a=mu(vo(r,"int32"),n),o=mu(vo(s,"int32"),n),i=ph(a),l=ai(i,o);return vo(l,"int32")}}),uf=Object.freeze(Object.defineProperty({__proto__:null,confusionMatrix:lf},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
let cf,hf=!1;function df(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(null==e)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,r=!1,s=!1,a=!1,o=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if("undefined"!=typeof ImageData&&e instanceof ImageData)r=!0;else if("undefined"!=typeof HTMLVideoElement&&e instanceof HTMLVideoElement)s=!0;else if("undefined"!=typeof HTMLImageElement&&e instanceof HTMLImageElement)a=!0;else if(null!=e.getContext)o=!0;else{if(!("undefined"!=typeof ImageBitmap&&e instanceof ImageBitmap))throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);i=!0}if(null!=gr(ir,ta.backendName)){const n={pixels:e},r={numChannels:t};return ta.runKernel(ir,n,r)}const[l,u]=s?[e.videoWidth,e.videoHeight]:[e.width,e.height];let c,h;if(o)c=e.getContext("2d").getImageData(0,0,l,u).data;else if(r||n)c=e.data;else if(a||s||i){if(null==cf)if("undefined"==typeof document){if("undefined"==typeof OffscreenCanvas||"undefined"==typeof OffscreenCanvasRenderingContext2D)throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");cf=new OffscreenCanvas(1,1).getContext("2d")}else cf=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});cf.canvas.width=l,cf.canvas.height=u,cf.drawImage(e,0,0,l,u),c=cf.getImageData(0,0,l,u).data}if(4===t)h=new Int32Array(c);else{const e=l*u;h=new Int32Array(e*t);for(let n=0;n<e;n++)for(let e=0;e<t;++e)h[n*t+e]=c[4*n+e]}return Jc(h,[u,l,t],"int32")}function ff(e){return"undefined"!=typeof window&&"undefined"!=typeof ImageBitmap&&window.hasOwnProperty("createImageBitmap")&&!(e instanceof ImageBitmap)&&function(e){return null!=e&&0!==e.width&&0!==e.height}(e)&&!function(e){return null!=e&&e.data instanceof Uint8Array}(e)}function pf(e){if(2!==e.rank&&3!==e.rank)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=2===e.rank?1:e.shape[2];if(t>4||2===t)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if("float32"!==e.dtype&&"int32"!==e.dtype)throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}async function gf(e,t){let n=ha(e,"img","toPixels");if(!(e instanceof Is)){const e=n;n=vo(e,"int32"),e.dispose()}pf(n);const[r,s]=n.shape.slice(0,2),a=2===n.rank?1:n.shape[2],o=await n.data(),i="float32"===n.dtype?255:1,l=new Uint8ClampedArray(s*r*4);for(let u=0;u<r*s;++u){const e=[0,0,0,255];for(let r=0;r<a;r++){const t=o[u*a+r];if("float32"===n.dtype){if(t<0||t>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${t}.`)}else if("int32"===n.dtype&&(t<0||t>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${t}.`);1===a?(e[0]=t*i,e[1]=t*i,e[2]=t*i):e[r]=t*i}const t=4*u;l[t+0]=Math.round(e[0]),l[t+1]=Math.round(e[1]),l[t+2]=Math.round(e[2]),l[t+3]=Math.round(e[3])}if(null!=t){if(!hf){null!=gr(Je,ta.backendName)&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),hf=!0)}t.width=s,t.height=r;const e=t.getContext("2d"),n=new ImageData(l,s,r);e.putImageData(n,0,0)}return n!==e&&n.dispose(),l}const mf=pa({fromPixels_:df}),bf=Object.freeze(Object.defineProperty({__proto__:null,draw:function(e,t,n){let r=ha(e,"img","draw");if(!(e instanceof Is)){const e=r;r=vo(e,"int32"),e.dispose()}pf(r),function(e){const t=(null==e?void 0:e.alpha)||1;if(t>1||t<0)throw new Error(`Alpha value ${t} is suppoed to be in range [0 - 1].`)}(null==n?void 0:n.imageOptions);const s={image:r},a={canvas:t,options:n};ta.runKernel(Je,s,a)},fromPixels:mf,fromPixelsAsync:async function(e,t=3){let n=null;if(J().getBool("WRAP_TO_IMAGEBITMAP")&&ff(e)){let t;try{t=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch(Lp){t=null}n=null!=t&&t.width===e.width&&t.height===e.height?t:e}else n=e;return df(n,t)},toPixels:gf},Symbol.toStringTag,{value:"Module"}));function yf(e,t){const n=e.shape.length,r=t.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if("int32"!==t.dtype)throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(0===d(e.shape))throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const s=t.shape,a=s[s.length-1];let o=1;for(let h=0;h<s.length-1;++h)o*=s[h];const i=e.shape,l=s.slice();l.pop();let u=1;for(let h=a;h<n;++h)u*=i[h],l.push(i[h]);const c=[...C(e.shape).map(e=>e/u),1].slice(0,a);return[l,o,u,c]}const wf=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:yf},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vf(e,t,n){const r=e.shape.length;u(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),u(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let s=0;s<r;++s)u(t[s]+n[s]<=e.shape[s],()=>`Error in slice${r}D: begin[${s}] + size[${s}] (${t[s]+n[s]}) would overflow input.shape[${s}] (${e.shape[s]})`)}function kf(e,t,n){const r=[];for(let s=0;s<e.length;s++)r[s]=Math.ceil((t[s]-e[s])/n[s]);return r}function xf(e,t,n,r){const s=[...e];for(let a=s.length;a<r.length;a++)s.push(1);for(let a=0;a<n;a++)0===a?s[t]=1:(s.splice(t,0,1),s.pop());return s}function Ef(e,t,n){return n<=e?n:n-(t-1)}function Sf(e,t){const n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function $f(e,t,n,r,s){const a=[...s],o=Sf(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{const s=Ef(t,n,i);let o=r[s];e&1<<s&&(o=0),a[i]=o}return a}function Af(e,t,n,r,s){const o=[...s],i=Sf(n,t);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=Number.MAX_SAFE_INTEGER;else{const s=Ef(t,n,a);let i=r[s];e&1<<s&&(i=Number.MAX_SAFE_INTEGER),o[a]=i}for(let l=0;l<o.length;l++){const e=s[l];o[l]<0&&(o[l]+=e),o[l]=a(0,o[l],s[l])}return o}function _f(e,t,n){let r=e[t];return(n&1<<t||null==r)&&(r=1),r}function If(e,t,n,r,s,o){let i=t[s];const l=n[s]||1;(e&1<<s||o&1<<s||null==i)&&(i=l>0?Number.MIN_SAFE_INTEGER:Number.MAX_SAFE_INTEGER);const u=r[s];return i<0&&(i+=u),i=a(0,i,u-1),i}function Mf(e,t,n,r,s,o){let i=t[s];const l=n[s]||1;(e&1<<s||o&1<<s||null==i)&&(i=l>0?Number.MAX_SAFE_INTEGER:Number.MIN_SAFE_INTEGER);const u=r[s];return i<0&&(i+=u),i=l>0?a(0,i,u):a(-1,i,u-1),i}function Tf(e,t,n){let r=n.length;for(let s=0;s<n.length;s++)if(n[s]>1){r=s;break}for(let s=r+1;s<n.length;s++)if(t[s]>0||n[s]!==e[s])return!1;return!0}function Nf(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function Df(e,t,n){let r;const s=e.shape.length;let a;return r="number"==typeof t?[t,...new Array(s-1).fill(0)]:t.length<s?t.concat(new Array(s-t.length).fill(0)):t.slice(),r.forEach(e=>{u(-1!==e,()=>"slice() does not support negative begin indexing.")}),a=null==n?new Array(s).fill(-1):"number"==typeof n?[n,...new Array(s-1).fill(-1)]:n.length<s?n.concat(new Array(s-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(u(-1===t,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function Rf(e,t,n,r,s,a,o,i,l){let u;if(null==r?(u=new Array(t.length),u.fill(1)):u=r,null!=o&&o&o-1)throw new Error("Multiple ellipses in slice is not allowed.");let c=!1;const h={dims:u.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:u.slice(),beginMask:s,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:l};for(let y=0;y<h.dims;y++)c&&1<<y&i&&h.numAddAxisAfterEllipsis++,1<<y&o&&(c=!0);c||(h.ellipsisMask|=1<<h.dims,h.dims++);const d={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};!function(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=null!=e.begin,t.endValid=null!=e.end,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){const s=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<s;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(-2),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);null!=e.begin&&(t.begin[n]=e.begin[r]),null!=e.end&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(-1),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}(h,d);let f=!0,p=!0,g=!0;const m=[],b=[];for(let y=0;y<e.length;++y){if(0===d.strides[y])throw Error(`strides[${y}] must be non-zero`);const t=!!(d.shrinkAxisMask&1<<y),n=e[y];if(-1===n){m.push(t?1:-1);continue}const r=[d.beginMask&1<<y,d.endMask&1<<y],s=[d.strides[y]>0?0:-1,d.strides[y]>0?n:n-1];if(t&&d.strides[y]<=0)throw Error("only stride 1 allowed on non-range indexing.");g=g&&1===d.strides[y];const a=!!(d.beginMask&1<<y&&d.endMask&1<<y);if(d.beginValid&&d.endValid){if(t){const e=d.begin[y]<0?n+d.begin[y]:d.begin[y];if(d.begin[y]=e,d.end[y]=d.begin[y]+1,e<0||e>=n)throw Error(`slice index ${d.begin[y]} of dimension ${y} out of bounds.`)}else d.begin[y]=Bf(d.begin[y],0,d.strides[y],n,r,s),d.end[y]=Bf(d.end[y],1,d.strides[y],n,r,s);const e=1===d.strides[y]&&0===d.begin[y]&&d.end[y]===n;f=f&&e,p=p&&(0===y&&1===d.strides[y]||e)}else f=f&&1===d.strides[y]&&a,p=p&&(0===y&&1===d.strides[y]||a);let o,i=!1;if(d.beginValid&&d.endValid?(o=d.end[y]-d.begin[y],i=!0):t?(o=1,i=!0):a&&n>=0&&(o=d.strides[y]<0?-n:n,i=!0),i){let e;e=0===o||o<0!=d.strides[y]<0?0:Math.trunc(o/d.strides[y])+(o%d.strides[y]!==0?1:0),m.push(e)}else m.push(-1)}for(let y=0;y<d.finalShapeGatherIndices.length;++y){const e=d.finalShapeGatherIndices[y];e>=0?b.push(m[e]):-2===e&&b.push(1)}return{finalShapeSparse:b.filter((e,t)=>-2!==d.finalShapeGatherIndices[t]),finalShape:b,isIdentity:f,sliceDim0:p,isSimpleSlice:g,begin:d.begin,end:d.end,strides:d.strides}}function Bf(e,t,n,r,s,a){if(s[t])return n>0?a[t]:a[t+1&1];{const t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}const Ff=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:vf,computeFlatOffset:Nf,computeOutShape:kf,getNormalizedAxes:function(e,t,n,r,s,a,o,i,l){const u=e.length;let c=new Array(u),h=new Array(u),d=new Array(u);if(t.length&&n>0){const l=t[0],u=n+1;c=$f(o,l,u,r,e),h=Af(i,l,u,s,e),d=xf(a,l,u,e)}else for(let f=0;f<u;f++)c[f]=If(o,r,a,e,f,l),h[f]=Mf(i,s,a,e,f,l),d[f]=_f(a,f,l);return{begin:c,end:h,strides:d}},isSliceContinous:Tf,maskToAxes:function(e){const t=[];let n=0;for(;e>0;)1&e&&t.push(n),e/=2,n++;return t},parseSliceParams:Df,sliceInfo:Rf,startForAxis:If,startIndicesWithElidedDims:$f,stopForAxis:Mf,stopIndicesWithElidedDims:Af,stridesForAxis:_f,stridesWithElidedDims:xf},Symbol.toStringTag,{value:"Module"}));
/** @license See the LICENSE file. */
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Cf{static sgd(e){return new Wd(e)}static momentum(e,t,n=!1){return new Gd(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,s=!1){return new Kd(e,t,n,r,s)}static adam(e=.001,t=.9,n=.999,r=null){return new Ld(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new Pd(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,s=0){return new Ud(e,t,n,r,s)}static adagrad(e,t=.1){return new zd(e,t)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Of=Cf,Pf="undefined"!=typeof requestAnimationFrame?requestAnimationFrame:"undefined"!=typeof setImmediate?setImmediate:e=>e();
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zf(){return new Promise(e=>Pf(()=>e()))}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lf(e,t){const n=e[0].length;e.forEach((e,t)=>{u(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),u(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const r=e[0];e.forEach((e,s)=>{for(let a=0;a<n;a++)u(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${s}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${s}.`)})}function Uf(e,t){const n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Wf,Gf;function Kf(e,t,n){let r=new Array;if(null==n&&null==t)return r;if(null==t)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(null==n)return r;if(e+n.length!==r.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let s=1;s<n.length;++s){const a=n[s],o=r[r.length-n.length+s],i=r[o];if(a>=0)if(i>=0){if(i!==a)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${s+e}] = ${a} but shape[${s+e}] = ${i}`)}else r[o]=a}return r}function qf(e){const t={FIRST_DIM_SIZE:Wf.FIRST_DIM_SIZE,VALUE_ROWIDS:Wf.VALUE_ROWIDS,ROW_LENGTHS:Wf.ROW_LENGTHS,ROW_SPLITS:Wf.ROW_SPLITS,ROW_LIMITS:Wf.ROW_LIMITS,ROW_STARTS:Wf.ROW_STARTS},n=[];for(const r of e){if(!(r in t))break;n.push(t[r])}return n}function jf(e){return 0===e.length?0:e[0]===Wf.FIRST_DIM_SIZE?e.length-1:e.length}function Vf(e,t){if(null==e||null==t)return;const n=e.length,r=t.length;if(n>=r)throw new Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let s=0;s<Math.min(n,r-1);++s){const n=e[s],r=t[s+1];if(n>=0&&r>=0&&1!==n&&n!==r)throw new Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${s-e.length}] = ${n} but ragged tensor input.flatValues.shape[${s-e.length}] = ${r}`)}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(Gf=Wf||(Wf={}))[Gf.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",Gf[Gf.VALUE_ROWIDS=1]="VALUE_ROWIDS",Gf[Gf.ROW_LENGTHS=2]="ROW_LENGTHS",Gf[Gf.ROW_SPLITS=3]="ROW_SPLITS",Gf[Gf.ROW_LIMITS=4]="ROW_LIMITS",Gf[Gf.ROW_STARTS=5]="ROW_STARTS";function Hf(e){return e<=30?e:F(e,Math.floor(Math.sqrt(e)))}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jf(e,t,n){return[n*("number"==typeof e?e:e[0]),t*("number"==typeof e?e:e[1])]}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zf(e,t,n,r=!0){let s=[];if(r)s=s.concat(t.slice(0)),s.push(e[0]/n),s=s.concat(e.slice(1));else{s=s.concat(e[0]);const n=t.length;for(let r=0;r<n;++r)s=s.concat([e[r+1]/t[r],t[r]]);s=s.concat(e.slice(n+1))}return s}function Yf(e,t,n=!0){const r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{const n=[],s=[];for(let r=1;r<e;++r)r>=2*t+1||r%2==1?s.push(r):n.push(r);r.push(...n),r.push(0),r.push(...s)}return r}function Xf(e,t,n,r=!0){const s=[];r?s.push(e[0]/n):s.push(e[0]*n);for(let a=1;a<e.length;++a)a<=t.length?r?s.push(t[a-1]*e[a]):s.push(e[a]/t[a-1]):s.push(e[a]);return s}function Qf(e,t){const n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function ep(e,t,n){const r=e.slice(0,1);for(let s=0;s<n;++s)r.push(e[s+1]-t[s][0]-t[s][1]);return r}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tp=1.7580993408473768,np=1.0507009873554805,rp=.3275911,sp=.254829592,ap=-.284496736,op=1.421413741,ip=-1.453152027,lp=1.061405429;
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function up(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const n=new Float32Array(2*e.length);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const cp="->",hp=/->/g;function dp(e,t){const n=((e=e.replace(/\s/g,"")).length-e.replace(hp,"").length)/2;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${cp}").`);const[r,s]=e.split(cp);u(-1===r.indexOf("..."),()=>'The ellipsis notation ("...") is not supported yet.');const a=r.split(","),o=a.length;if(t!==o)throw new Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const i=[];for(let u=0;u<s.length;++u){const e=s[u];if(!a.some(t=>-1!==t.indexOf(e)))throw new Error(`Output subscripts contain the label ${e} not present in the input subscripts.`);-1===i.indexOf(e)&&i.push(e)}for(let u=0;u<r.length;++u){const e=r[u];-1===i.indexOf(e)&&","!==e&&i.push(e)}const l=new Array(a.length);for(let u=0;u<o;++u){if(new Set(a[u].split("")).size!==a[u].length)throw new Error(`Found duplicate axes in input component ${a[u]}. Support for duplicate axes in input is not implemented yet.`);l[u]=[];for(let e=0;e<a[u].length;++e)l[u].push(i.indexOf(a[u][e]))}const c=i.length,h=[];for(let u=s.length;u<c;++u)h.push(u);return{allDims:i,summedDims:h,idDims:l}}function fp(e,t){let n=new Array(e);n.fill(-1);for(let s=0;s<t.length;++s)n[t[s]]=s;const r=[];for(let s=0;s<e;++s)-1===n[s]&&r.push(s);return n=n.filter(e=>-1!==e),{permutationIndices:n,expandDims:r}}function pp(e,t,n){const r=new Array(e);for(let s=0;s<n.length;++s){const e=n[s].shape;for(let n=0;n<t[s].length;++n)void 0===r[t[s][n]]?r[t[s][n]]=e[n]:u(r[t[s][n]]===e[n],()=>`Expected dimension ${r[t[s][n]]} at axis ${n} of input shaped ${JSON.stringify(e)}, but got dimension ${e[n]}`)}}function gp(e,t){const n=e,r=[];let s=0;0===e.length&&n.push(-1),s=e.length+1;for(let o=0;o<s;++o)r.push([]);const a=[];for(let o=0;o<n.length;++o){const e=bp(t,n[o]);for(const t of e)-1===a.indexOf(t)&&(r[o].push(t),a.push(t))}return{path:n,steps:r}}function mp(e){return e.every((e,t)=>e===t)}function bp(e,t){const n=[];for(let r=0;r<e.length;++r)0!==e[r].length&&-1===e[r].indexOf(t)&&-1!==t||n.push(r);return n}function yp(e,t,n=0){let r=[];if("number"==typeof t)u(e.shape[n]%t===0,()=>"Number of splits must evenly divide the axis."),r=new Array(t).fill(e.shape[n]/t);else{u(t.reduce((e,t)=>(-1===t&&(e+=1),e),0)<=1,()=>"There should be only one negative value in split array.");const s=t.indexOf(-1);if(-1!==s){const r=t.reduce((e,t)=>t>0?e+t:e);t[s]=e.shape[n]-r}u(e.shape[n]===t.reduce((e,t)=>e+t),()=>"The sum of sizes must match the size of the axis dimension."),r=t}return r}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wp(e){return`Received SparseTensor with denseShape[0] = 0 but\n  indices.shape[0] = ${e}`}function vp(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function kp(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xp(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function Ep(e,t){return`size ${e} must be non-negative, not ${t}`}function Sp(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function $p(e,t){return`Input to reshape is a SparseTensor with ${d(e)}\n  dense values, but the requested shape requires a multiple of ${d(t)}. inputShape=${e} outputShape= ${t}`}function Ap(e,t){return`Input to reshape is a tensor with ${d(e)} dense values, but the requested shape has ${d(t)}. inputShape=${e} outputShape=${t}`}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _p(){return"segment ids must be >= 0"}function Ip(){return"segment ids are not increasing"}function Mp(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function Tp(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Np(e,t){let n,r=!1;for(e<=30?(n=e,r=!0):n=F(e,Math.floor(Math.sqrt(e)));!r;)n>t||n===e?r=!0:n=F(e,n+1);return n}function Dp(e,t,n){const r=[],s=e.length;for(let a=0;a<s;a++)a!==t?r.push(e[a]):r.push(n);return r}function Rp(e,t,n,r){const s=t.shape.length,a=e.shape.length;if(0!==r&&(r<-s||r>s))throw new Error(`Expect batchDims in the range of [-${s}, ${s}], but got ${r}`);if(r<0&&(r+=s),r>a)throw new Error(`batchDims (${r}) must be less than rank(x) (\n    ${a}).`);if(n<r)throw new Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let h=0;h<r;++h)if(e.shape[h]!==t.shape[h])throw new Error(`x.shape[${h}]: ${e.shape[h]} should be equal to indices.shape[${h}]: ${t.shape[h]}.`);const o=e.shape[n],i=[];let l=1,u=1,c=1;for(let h=0;h<r;++h)i.push(e.shape[h]),l*=e.shape[h];for(let h=r;h<n;h++)i.push(e.shape[h]),u*=e.shape[h];for(let h=r;h<s;h++)i.push(t.shape[h]);for(let h=n+1;h<a;h++)i.push(e.shape[h]),c*=e.shape[h];return{batchSize:l,sliceSize:c,outerSize:u,dimSize:o,outputShape:i}}const Bp=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:Rp,computeOutShape:Dp,segOpComputeOptimalWindowSize:Np},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fp(e){try{return e.map(e=>fs(e))}catch(Up){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${Up}`)}}function Cp(e){return e.map(e=>ds(e))}const Op=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:sp,ERF_A2:ap,ERF_A3:op,ERF_A4:ip,ERF_A5:lp,ERF_P:rp,PARALLELIZE_THRESHOLD:30,get RowPartitionType(){return Wf},SELU_SCALE:np,SELU_SCALEALPHA:tp,applyActivation:Ah,assertAndGetBroadcastShape:Ki,assertAxesAreInnerMostDims:al,assertParamsConsistent:Lf,assignToTypedArray:function(e,t,n,r){e[2*r]=t,e[2*r+1]=n},axesAreInnerMostDims:tl,calculateShapes:th,checkEinsumDimSizes:pp,checkPadOnDimRoundingMode:ei,combineLocations:nl,combineRaggedTensorToTensorShapes:Kf,complexWithEvenIndex:function(e){const t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=0;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}},complexWithOddIndex:function(e){const t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=2;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}},computeConv2DInfo:Go,computeConv3DInfo:Ko,computeDefaultPad:qo,computeDilation2DInfo:Lo,computeOptimalWindowSize:Hf,computeOutAndReduceShapes:rl,computeOutShape:Uf,computePool2DInfo:Uo,computePool3DInfo:Wo,convertConv2DDataFormat:Qo,decodeEinsumEquation:dp,eitherStridesOrDilationsAreOne:Yo,expandShapeToKeepDim:sl,exponent:function(e,t,n){const r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}},exponents:function(e,t){const n=new Float32Array(e/2),r=new Float32Array(e/2);for(let s=0;s<Math.ceil(e/2);s++){const a=(t?2:-2)*Math.PI*(s/e);n[s]=Math.cos(a),r[s]=Math.sin(a)}return{real:n,imag:r}},fromStringArrayToUint8:Cp,fromUint8ToStringArray:Fp,getAxesPermutation:ol,getBroadcastDims:Wi,getComplexWithIndex:function(e,t){return{real:e[2*t],imag:e[2*t+1]}},getEinsumComputePath:gp,getEinsumPermutation:fp,getFusedBiasGradient:$h,getFusedDyActivation:Sh,getImageCenter:Jf,getInnerMostAxes:ll,getPermuted:Yf,getRaggedRank:jf,getReductionAxes:Gi,getReshaped:Zf,getReshapedPermuted:Xf,getRowPartitionTypesHelper:qf,getSliceBeginCoords:Qf,getSliceSize:ep,getSparseFillEmptyRowsIndicesDenseShapeMismatch:wp,getSparseFillEmptyRowsNegativeIndexErrorMessage:vp,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:kp,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:Sp,getSparseReshapeInputOutputMismatchErrorMessage:Ap,getSparseReshapeInputOutputMultipleErrorMessage:$p,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:xp,getSparseReshapeNegativeOutputDimErrorMessage:Ep,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Tp,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:_p,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Ip,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Mp,getUndoAxesPermutation:il,isIdentityPermutation:mp,log:function(...e){J().getBool("IS_TEST")||J().getBool("PROD")||console.log(...e)},mergeRealAndImagArrays:up,prepareAndValidate:yf,prepareSplitSize:yp,segment_util:Bp,shouldFuse:_h,slice_util:Ff,splitRealAndImagArrays:function(e){const t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}},stridesOrDilationsArePositive:Xo,tupleValuesAreOne:Zo,upcastType:Ws,validateDefaultValueShape:Vf,validateInput:eh,validateUpdateShape:Qc,warn:dr},Symbol.toStringTag,{value:"Module"})),Pp=Object.freeze(Object.defineProperty({__proto__:null,nonMaxSuppressionV3Impl:Vh,nonMaxSuppressionV4Impl:Hh,nonMaxSuppressionV5Impl:Jh,whereImpl:hh},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
!function(){for(const e of qd)Fd(e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */();const zp=Object.freeze(Object.defineProperty({__proto__:null,Abs:ee,Acos:te,Acosh:ne,AdadeltaOptimizer:Pd,AdagradOptimizer:zd,AdamOptimizer:Ld,AdamaxOptimizer:Ud,Add:re,AddN:se,All:ae,Any:oe,ArgMax:ie,ArgMin:le,Asin:ue,Asinh:ce,Atan:he,Atan2:fe,Atanh:de,AvgPool:pe,AvgPool3D:me,AvgPool3DGrad:be,AvgPoolGrad:ge,BatchMatMul:ye,BatchToSpaceND:we,Bincount:ve,BitwiseAnd:ke,BroadcastArgs:xe,BroadcastTo:"BroadcastTo",Cast:Ee,Ceil:Se,ClipByValue:$e,Complex:Ae,ComplexAbs:_e,Concat:Ie,Conv2D:Me,Conv2DBackpropFilter:Te,Conv2DBackpropInput:Ne,Conv3D:De,Conv3DBackpropFilterV2:Re,Conv3DBackpropInputV2:Be,Cos:Fe,Cosh:Ce,CropAndResize:ze,Cumprod:Oe,Cumsum:Pe,DataStorage:t,DenseBincount:Le,DepthToSpace:Ue,DepthwiseConv2dNative:We,DepthwiseConv2dNativeBackpropFilter:Ge,DepthwiseConv2dNativeBackpropInput:Ke,Diag:qe,Dilation2D:je,Dilation2DBackpropFilter:He,Dilation2DBackpropInput:Ve,Draw:Je,get ENV(){return Y},Einsum:Ye,Elu:Xe,EluGrad:Qe,Environment:V,Equal:tt,Erf:et,Exp:nt,ExpandDims:rt,Expm1:st,FFT:at,Fill:ot,FlipLeftRight:it,Floor:lt,FloorDiv:ut,FromPixels:ir,FusedBatchNorm:ct,FusedConv2D:cr,FusedDepthwiseConv2D:hr,GatherNd:dt,GatherV2:ht,Greater:ft,GreaterEqual:pt,IFFT:mt,Identity:gt,Imag:bt,IsFinite:yt,IsInf:wt,IsNan:vt,KernelBackend:n,LRN:Tt,LRNGrad:Nt,LeakyRelu:kt,Less:xt,LessEqual:Et,LinSpace:St,Log:$t,Log1p:At,LogSoftmax:"LogSoftmax",LogicalAnd:_t,LogicalNot:It,LogicalOr:Mt,LogicalXor:"LogicalXor",LowerBound:"LowerBound",MatrixBandPart:"MatrixBandPart",Max:Dt,MaxPool:Bt,MaxPool3D:Ct,MaxPool3DGrad:Ot,MaxPoolGrad:Ft,MaxPoolWithArgmax:Pt,Maximum:Rt,Mean:zt,Min:Lt,Minimum:Ut,MirrorPad:Wt,Mod:Gt,MomentumOptimizer:Gd,Multinomial:Kt,Multiply:qt,Neg:jt,NonMaxSuppressionV3:Ht,NonMaxSuppressionV4:Jt,NonMaxSuppressionV5:Zt,NotEqual:Vt,OP_SCOPE_SUFFIX:fa,OneHot:Xt,OnesLike:Yt,Optimizer:Od,OptimizerConstructors:Cf,Pack:Qt,PadV2:en,Pool:"Pool",Pow:tn,Prelu:nn,Prod:rn,RMSPropOptimizer:Kd,RaggedGather:sn,RaggedRange:an,RaggedTensorToTensor:on,Range:ln,get Rank(){return Ns},Real:un,RealDiv:Ze,Reciprocal:cn,get Reduction(){return fd},Relu:hn,Relu6:bn,Reshape:dn,ResizeBilinear:gn,ResizeBilinearGrad:mn,ResizeNearestNeighbor:fn,ResizeNearestNeighborGrad:pn,Reverse:yn,RotateWithOffset:lr,Round:wn,Rsqrt:vn,SGDOptimizer:Wd,ScatterNd:kn,SearchSorted:En,Select:Sn,Selu:$n,Sigmoid:Tn,Sign:Mn,Sin:_n,Sinh:In,Slice:An,Softmax:Cn,Softplus:Nn,SpaceToBatchND:Bn,SparseFillEmptyRows:On,SparseReshape:Pn,SparseSegmentMean:zn,SparseSegmentSum:Ln,SparseToDense:Un,SplitV:Fn,Sqrt:Dn,Square:Gn,SquaredDifference:Wn,StaticRegexReplace:Kn,Step:or,StridedSlice:qn,StringNGrams:jn,StringSplit:Vn,StringToHashBucketFast:Hn,Sub:Jn,Sum:Rn,Tan:Zn,Tanh:Yn,Tensor:Is,TensorBuffer:$s,TensorScatterUpdate:xn,Tile:Xn,TopK:Qn,Transform:er,Transpose:tr,Unique:nr,Unpack:rr,UnsortedSegmentSum:sr,UpperBound:"UpperBound",Variable:Ts,ZerosLike:ar,_FusedMatMul:ur,abs:_o,acos:Io,acosh:Mo,add:Eo,addN:To,all:No,any:Do,argMax:Ro,argMin:Bo,asin:Fo,asinh:Co,atan:Oo,atan2:Po,atanh:zo,avgPool:ni,avgPool3d:ri,backend:Aa,backend_util:Op,basicLSTMCell:ui,batchNorm:hi,batchNorm2d:di,batchNorm3d:fi,batchNorm4d:pi,batchToSpaceND:ci,bincount:gi,bitwiseAnd:mi,booleanMaskAsync:fh,broadcastArgs:bi,broadcastTo:yi,broadcast_util:qi,browser:bf,buffer:wo,cast:vo,ceil:wi,clipByValue:ki,clone:ko,complex:ga,concat:si,concat1d:xi,concat2d:Ei,concat3d:Si,concat4d:$i,conv1d:_i,conv2d:Ai,conv2dTranspose:Mi,conv3d:Ti,conv3dTranspose:Di,copyRegisteredKernels:function(e,t){br(e).forEach(e=>{yr(Object.assign({},e,{backendName:t}))})},cos:Ri,cosh:Bi,cosineWindow:kh,cumprod:Fi,cumsum:Ci,customGrad:Ll,denseBincount:Oi,deprecationWarn:function(e){J().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")},depthToSpace:Pi,depthwiseConv2d:zi,device_util:oa,diag:Li,dilation2d:Ui,disableDeprecationWarnings:function(){J().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")},dispose:xa,disposeVariables:function(){ta.disposeVariables()},div:$o,divNoNan:Ji,dot:Zi,dropout:wh,einsum:Yi,elu:Xi,enableDebugMode:function(){J().set("DEBUG",!0)},enableProdMode:function(){J().set("PROD",!0)},enclosingPowerOfTwo:vh,engine:va,ensureShape:Qi,env:J,equal:ji,erf:el,euclideanNorm:yl,exp:wl,expandDims:vl,expm1:kl,eye:El,fft:Cc,fill:vi,findBackend:function(e){return ta.findBackend(e)},findBackendFactory:function(e){return ta.findBackendFactory(e)},floor:Sl,floorDiv:So,fused:Rh,gather:$l,gatherND:yh,gather_util:wf,getBackend:Sa,getGradient:mr,getKernel:gr,getKernelsForBackend:br,grad:function(e){return u(B(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const r=ha(t,"x","tf.grad","string_or_numeric"),s=null!=n?ha(n,"dy","tf.grad"):null;return ta.tidy(()=>{const{value:t,grads:n}=ta.gradients(()=>e(r),[r],s);return null!=s&&c(t.shape,s.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Ul(n),n[0]})}},grads:function(e){return u(B(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{u(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const r=da(t,"args","tf.grads","string_or_numeric"),s=null!=n?ha(n,"dy","tf.grads"):null;return ta.tidy(()=>{const{value:t,grads:n}=ta.gradients(()=>e(...r),r,s);return null!=s&&c(t.shape,s.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Ul(n),n})}},greater:Al,greaterEqual:_l,ifft:Oc,imag:Il,image:Ad,inTopKAsync:xh,io:of,irfft:Pc,isFinite:Ml,isInf:Tl,isNaN:Nl,keep:Ea,kernel_impls:Pp,leakyRelu:Dl,less:Rl,lessEqual:Bl,linalg:_d,linspace:Fl,localResponseNormalization:Cl,log:Ol,log1p:Pl,logSigmoid:Kl,logSoftmax:jl,logSumExp:Vl,logicalAnd:Hl,logicalNot:Jl,logicalOr:Zl,logicalXor:Yl,losses:Id,lowerBound:eu,matMul:ai,math:uf,max:ul,maxPool:tu,maxPool3d:nu,maxPoolWithArgmax:ru,maximum:su,mean:au,memory:function(){return ta.memory()},meshgrid:lu,min:cl,minimum:uu,mirrorPad:cu,mod:hu,moments:du,movingAverage:gh,mul:Ao,multiRNNCell:fu,multinomial:pu,neg:Wl,nextFrame:zf,norm:bl,notEqual:gu,oneHot:mu,ones:iu,onesLike:bu,op:pa,outerProduct:yu,pad:wu,pad1d:vu,pad2d:ku,pad3d:xu,pad4d:Eu,pool:$u,pow:hl,prelu:Au,print:xo,prod:_u,profile:function(e){return ta.profile(e)},raggedGather:Iu,raggedRange:Mu,raggedTensorToTensor:Tu,rand:Nu,randomGamma:lc,randomNormal:uc,randomStandardNormal:cc,randomUniform:hc,randomUniformInt:dc,range:fc,ready:function(){return ta.ready()},real:pc,reciprocal:gc,registerBackend:$a,registerGradient:function(e){const{kernelName:t}=e;pr.has(t)&&J().getBool("DEBUG")&&dr(`Overriding the gradient for '${t}'`),pr.set(t,e)},registerKernel:yr,relu:mc,relu6:bc,removeBackend:function(e){ta.removeBackend(e)},reshape:ti,reverse:yc,reverse1d:wc,reverse2d:vc,reverse3d:kc,reverse4d:xc,rfft:Lc,round:Ec,rsqrt:Sc,scalar:dl,scatterND:mh,scatter_util:nh,searchSorted:Ql,selu:$c,separableConv2d:Ac,serialization:Cd,setBackend:function(e){return ta.setBackend(e)},setPlatform:function(e,t){J().setPlatform(e,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */,setdiff1dAsync:_c,sigmoid:oi,sign:Ic,signal:$d,sin:Mc,sinh:Tc,slice:ii,slice1d:Nc,slice2d:Dc,slice3d:Rc,slice4d:Bc,slice_util:Ff,softmax:Fc,softplus:Gl,spaceToBatchND:Su,sparse:Md,sparseToDense:bh,spectral:Sd,split:zc,sqrt:fl,square:pl,squaredDifference:Uc,squeeze:Wc,stack:Gc,step:Kc,stridedSlice:qc,string:Td,sub:ql,sum:gl,sumOutType:Gs,tan:jc,tanh:li,tensor:ba,tensor1d:Vc,tensor2d:Hc,tensor3d:Jc,tensor4d:Zc,tensor5d:Yc,tensor6d:Xc,tensorScatterUpdate:rh,tensor_util:Zs,test_util:sc,tidy:ka,tile:xl,time:function(e){return ta.time(e)},topk:sh,train:Of,transpose:ph,truncatedNormal:ah,unique:oh,unregisterGradient:function(e){if(!pr.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);pr.delete(e)},unregisterKernel:function(e,t){const n=wr(e,t);if(!fr.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);fr.delete(n)},unsortedSegmentSum:ih,unstack:lh,upcastType:Ws,upperBound:uh,util:ms,valueAndGrad:function(e){return u(B(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{u(t instanceof Is,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),u(null==n||n instanceof Is,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:r,value:s}=ta.gradients(()=>e(t),[t],n);return Ul(r),{grad:r[0],value:s}}},valueAndGrads:function(e){return u(B(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{u(Array.isArray(t)&&t.every(e=>e instanceof Is),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),u(null==n||n instanceof Is,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const r=ta.gradients(()=>e(...t),t,n);return null!=n&&c(r.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Ul(r.grads),r}},variable:ch,variableGrads:zl,version_core:"4.22.0",where:Vi,whereAsync:dh,zeros:ou,zerosLike:Hi},Symbol.toStringTag,{value:"Module"}));export{Rn as $,I as A,zf as B,hh as C,t as D,aa as E,$a as F,Ki as G,Ae as H,gt as I,us as J,n as K,kt as L,Ws as M,Fp as N,qt as O,nn as P,v as Q,dn as R,m as S,Hf as T,k as U,ol as V,ll as W,al as X,rl as Y,sl as Z,Gs as _,b as a,il as a$,tr as a0,ur as a1,ee as a2,te as a3,ne as a4,re as a5,se as a6,ae as a7,oe as a8,ie as a9,ve as aA,ke as aB,xe as aC,Vt as aD,un as aE,Ee as aF,ou as aG,_ as aH,Se as aI,$e as aJ,_e as aK,Uf as aL,bt as aM,Ie as aN,Lf as aO,Me as aP,Qo as aQ,Go as aR,Te as aS,Ne as aT,De as aU,Ko as aV,Re as aW,Be as aX,Fe as aY,Ce as aZ,ze as a_,le as aa,ue as ab,ce as ac,he as ad,fe as ae,de as af,pe as ag,Yo as ah,Uo as ai,me as aj,Wo as ak,be as al,ge as am,ye as an,ct as ao,An as ap,Df as aq,vf as ar,Tf as as,Nf as at,we as au,Zf as av,Yf as aw,Xf as ax,Qf as ay,ep as az,u as b,Ot as b$,Oe as b0,Pe as b1,Le as b2,Ue as b3,We as b4,Ge as b5,Ke as b6,qe as b7,je as b8,Lo as b9,ir as bA,cr as bB,hr as bC,dt as bD,yf as bE,ht as bF,Rp as bG,ft as bH,pt as bI,mt as bJ,yt as bK,wt as bL,vt as bM,xt as bN,Et as bO,St as bP,$t as bQ,At as bR,_t as bS,It as bT,Mt as bU,Tt as bV,Nt as bW,Dt as bX,Rt as bY,Bt as bZ,Ct as b_,Ye as ba,dp as bb,pp as bc,gp as bd,fp as be,mp as bf,Xe as bg,Qe as bh,tt as bi,et as bj,rp as bk,sp as bl,ap as bm,op as bn,ip as bo,lp as bp,nt as bq,rt as br,st as bs,at as bt,ot as bu,R as bv,S as bw,it as bx,lt as by,ut as bz,x as c,Gn as c$,Ft as c0,Pt as c1,zt as c2,Lt as c3,Ut as c4,Wt as c5,Gt as c6,Ze as c7,Jn as c8,Cn as c9,pn as cA,yn as cB,lr as cC,Jf as cD,wn as cE,vn as cF,kn as cG,th as cH,En as cI,Sn as cJ,$n as cK,tp as cL,np as cM,Tn as cN,Mn as cO,_n as cP,In as cQ,Nn as cR,Bn as cS,On as cT,Pn as cU,zn as cV,Ln as cW,Un as cX,Fn as cY,yp as cZ,Dn as c_,Kt as ca,jt as cb,Ht as cc,Vh as cd,Jt as ce,Hh as cf,Zt as cg,Jh as ch,Xt as ci,ar as cj,Yt as ck,Qt as cl,c as cm,en as cn,tn as co,rn as cp,sn as cq,an as cr,on as cs,ln as ct,cn as cu,hn as cv,bn as cw,gn as cx,mn as cy,fn as cz,g as d,Ai as d$,Wn as d0,Kn as d1,or as d2,qn as d3,Rf as d4,kf as d5,jn as d6,Vn as d7,Hn as d8,Zn as d9,Co as dA,Oo as dB,Po as dC,zo as dD,ni as dE,ri as dF,ui as dG,hi as dH,di as dI,fi as dJ,pi as dK,ci as dL,gi as dM,mi as dN,fh as dO,bi as dP,yi as dQ,vo as dR,wi as dS,ki as dT,ga as dU,si as dV,xi as dW,Ei as dX,Si as dY,$i as dZ,_i as d_,Yn as da,xn as db,Xn as dc,Qn as dd,er as de,nr as df,rr as dg,sr as dh,Dp as di,Np as dj,yr as dk,kr as dl,P as dm,ko as dn,fa as dp,_o as dq,Io as dr,Mo as ds,Eo as dt,To as du,No as dv,Do as dw,Ro as dx,Bo as dy,Fo as dz,J as e,ul as e$,Mi as e0,Ti as e1,Di as e2,Ri as e3,Bi as e4,kh as e5,Fi as e6,Ci as e7,Oi as e8,Pi as e9,_l as eA,Oc as eB,Il as eC,Ad as eD,xh as eE,Pc as eF,Ml as eG,Tl as eH,Nl as eI,Dl as eJ,Rl as eK,Bl as eL,_d as eM,Fl as eN,Cl as eO,Ol as eP,Pl as eQ,Kl as eR,jl as eS,Vl as eT,Hl as eU,Jl as eV,Zl as eW,Yl as eX,Id as eY,eu as eZ,ai as e_,zi as ea,Li as eb,Ui as ec,$o as ed,Ji as ee,Zi as ef,wh as eg,Yi as eh,Xi as ei,vh as ej,Qi as ek,ji as el,el as em,yl as en,wl as eo,vl as ep,kl as eq,El as er,Cc as es,vi as et,Sl as eu,So as ev,Rh as ew,$l as ex,yh as ey,Al as ez,C as f,$d as f$,tu as f0,nu as f1,ru as f2,su as f3,au as f4,lu as f5,cl as f6,uu as f7,cu as f8,hu as f9,Nu as fA,lc as fB,uc as fC,cc as fD,hc as fE,dc as fF,fc as fG,pc as fH,gc as fI,mc as fJ,bc as fK,ti as fL,yc as fM,wc as fN,vc as fO,kc as fP,xc as fQ,Lc as fR,Ec as fS,Sc as fT,mh as fU,Ql as fV,$c as fW,Ac as fX,_c as fY,oi as fZ,Ic as f_,du as fa,gh as fb,Ao as fc,fu as fd,pu as fe,Wl as ff,bl as fg,gu as fh,mu as fi,iu as fj,bu as fk,pa as fl,yu as fm,wu as fn,vu as fo,ku as fp,xu as fq,Eu as fr,$u as fs,hl as ft,Au as fu,xo as fv,_u as fw,Iu as fx,Mu as fy,Tu as fz,Op as g,G as g$,Mc as g0,Tc as g1,ii as g2,Nc as g3,Dc as g4,Rc as g5,Bc as g6,Fc as g7,Gl as g8,Su as g9,ph as gA,ah as gB,oh as gC,ih as gD,lh as gE,uh as gF,ch as gG,Vi as gH,dh as gI,Hi as gJ,Ea as gK,q as gL,of as gM,Da as gN,Is as gO,xa as gP,z as gQ,f as gR,He as gS,Ve as gT,Je as gU,U as gV,Sa as gW,gf as gX,mf as gY,Aa as gZ,K as g_,Md as ga,bh as gb,Sd as gc,zc as gd,fl as ge,pl as gf,Uc as gg,Wc as gh,Gc as gi,Kc as gj,qc as gk,Td as gl,ql as gm,gl as gn,jc as go,li as gp,ba as gq,Vc as gr,Hc as gs,Jc as gt,Zc as gu,Yc as gv,Xc as gw,rh as gx,xl as gy,sh as gz,Wi as h,cs as h0,qf as h1,jf as h2,Wf as h3,Vf as h4,Kf as h5,$s as h6,Cp as h7,wp as h8,vp as h9,kp as ha,xp as hb,Ep as hc,Sp as hd,$p as he,Ap as hf,_p as hg,Ip as hh,Mp as hi,Tp as hj,ls as hk,i as hl,zp as hm,sa as i,w as j,va as k,hs as l,up as m,o as n,fs as o,wo as p,gs as q,y as r,d as s,l as t,T as u,ds as v,dr as w,E as x,ka as y,dl as z};
