/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TinyFusionOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::TinyFusion::Conv2dLReluOp,
::mlir::TinyFusion::Conv2dReluOp,
::mlir::TinyFusion::Conv2dSiluOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TinyFusion {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TinyFusionOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TinyFusionOps0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TinyFusionOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TinyFusionOps0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TinyFusionOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::FloatAttr>(attr))) && ((::llvm::cast<::mlir::FloatAttr>(attr).getType().isF32()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TinyFusionOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TinyFusionOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace TinyFusion
} // namespace mlir
namespace mlir {
namespace TinyFusion {

//===----------------------------------------------------------------------===//
// ::mlir::TinyFusion::Conv2dLReluOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2dLReluOpGenericAdaptorBase::Conv2dLReluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("TinyFusion.conv2d_lrelu", odsAttrs.getContext());
}

Conv2dLReluOpGenericAdaptorBase::Conv2dLReluOpGenericAdaptorBase(Conv2dLReluOp op) : Conv2dLReluOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::ArrayAttr Conv2dLReluOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dLReluOpGenericAdaptorBase::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dLReluOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

} // namespace detail
Conv2dLReluOpAdaptor::Conv2dLReluOpAdaptor(Conv2dLReluOp op) : Conv2dLReluOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult Conv2dLReluOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""requires attribute 'dilation'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""requires attribute 'stride'");

  if (tblgen_dilation && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_dilation))))
    return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""attribute 'dilation' failed to satisfy constraint: array attribute");

  if (tblgen_padding && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_padding))))
    return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""attribute 'padding' failed to satisfy constraint: array attribute");

  if (tblgen_stride && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_stride))))
    return emitError(loc, "'TinyFusion.conv2d_lrelu' op ""attribute 'stride' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

::mlir::LogicalResult Conv2dLReluOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.dilation;
       auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dilation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.padding;
       auto attr = dict.get("padding");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for padding in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `padding` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
       auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `stride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute Conv2dLReluOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.padding;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("padding",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code Conv2dLReluOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.padding.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> Conv2dLReluOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "padding")
      return prop.padding;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void Conv2dLReluOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "padding") {
       prop.padding = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.padding)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void Conv2dLReluOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.padding) attrs.append("padding", prop.padding);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult Conv2dLReluOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "dilation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPaddingAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "padding", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "stride", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult Conv2dLReluOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.dilation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.padding)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.stride)))
    return ::mlir::failure();
  return ::mlir::success();
}

void Conv2dLReluOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.dilation);
  writer.writeAttribute(prop.padding);
  writer.writeAttribute(prop.stride);
}

::mlir::ArrayAttr Conv2dLReluOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dLReluOp::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dLReluOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

void Conv2dLReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, ::mlir::Value bias_tensor, ::mlir::Value neg_slop, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  odsState.addOperands(bias_tensor);
  odsState.addOperands(neg_slop);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.addTypes(result);
}

void Conv2dLReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, ::mlir::Value bias_tensor, ::mlir::Value neg_slop, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  odsState.addOperands(bias_tensor);
  odsState.addOperands(neg_slop);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2dLReluOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv2dLReluOp::verifyInvariantsImpl() {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitOpError("requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_padding, "padding")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_stride, "stride")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2dLReluOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult Conv2dLReluOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand input_tensorRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> input_tensorOperands(&input_tensorRawOperand, 1);  ::llvm::SMLoc input_tensorOperandsLoc;
  (void)input_tensorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand weight_tensorRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> weight_tensorOperands(&weight_tensorRawOperand, 1);  ::llvm::SMLoc weight_tensorOperandsLoc;
  (void)weight_tensorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bias_tensorRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bias_tensorOperands(&bias_tensorRawOperand, 1);  ::llvm::SMLoc bias_tensorOperandsLoc;
  (void)bias_tensorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand neg_slopRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> neg_slopOperands(&neg_slopRawOperand, 1);  ::llvm::SMLoc neg_slopOperandsLoc;
  (void)neg_slopOperandsLoc;
  ::mlir::Type input_tensorRawType{};
  ::llvm::ArrayRef<::mlir::Type> input_tensorTypes(&input_tensorRawType, 1);
  ::mlir::Type weight_tensorRawType{};
  ::llvm::ArrayRef<::mlir::Type> weight_tensorTypes(&weight_tensorRawType, 1);
  ::mlir::Type bias_tensorRawType{};
  ::llvm::ArrayRef<::mlir::Type> bias_tensorTypes(&bias_tensorRawType, 1);
  ::mlir::Type neg_slopRawType{};
  ::llvm::ArrayRef<::mlir::Type> neg_slopTypes(&neg_slopRawType, 1);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  input_tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input_tensorRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  weight_tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(weight_tensorRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bias_tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bias_tensorRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  neg_slopOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(neg_slopRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    input_tensorRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    weight_tensorRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bias_tensorRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    neg_slopRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(input_tensorOperands, input_tensorTypes, input_tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(weight_tensorOperands, weight_tensorTypes, weight_tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bias_tensorOperands, bias_tensorTypes, bias_tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(neg_slopOperands, neg_slopTypes, neg_slopOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void Conv2dLReluOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputTensor();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getWeightTensor();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBiasTensor();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getNegSlop();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getInputTensor().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getWeightTensor().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getBiasTensor().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getNegSlop().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void Conv2dLReluOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace TinyFusion
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::Conv2dLReluOp)

namespace mlir {
namespace TinyFusion {

//===----------------------------------------------------------------------===//
// ::mlir::TinyFusion::Conv2dReluOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2dReluOpGenericAdaptorBase::Conv2dReluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("TinyFusion.conv2d_relu", odsAttrs.getContext());
}

Conv2dReluOpGenericAdaptorBase::Conv2dReluOpGenericAdaptorBase(Conv2dReluOp op) : Conv2dReluOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> Conv2dReluOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::llvm::APFloat Conv2dReluOpGenericAdaptorBase::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

::llvm::APFloat Conv2dReluOpGenericAdaptorBase::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

} // namespace detail
Conv2dReluOpAdaptor::Conv2dReluOpAdaptor(Conv2dReluOp op) : Conv2dReluOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult Conv2dReluOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'TinyFusion.conv2d_relu' op ""requires attribute 'dilation'");
  auto tblgen_max_fp = getProperties().max_fp; (void)tblgen_max_fp;
  if (!tblgen_max_fp) return emitError(loc, "'TinyFusion.conv2d_relu' op ""requires attribute 'max_fp'");
  auto tblgen_min_fp = getProperties().min_fp; (void)tblgen_min_fp;
  if (!tblgen_min_fp) return emitError(loc, "'TinyFusion.conv2d_relu' op ""requires attribute 'min_fp'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitError(loc, "'TinyFusion.conv2d_relu' op ""requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'TinyFusion.conv2d_relu' op ""requires attribute 'stride'");

  if (tblgen_dilation && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_dilation))))
    return emitError(loc, "'TinyFusion.conv2d_relu' op ""attribute 'dilation' failed to satisfy constraint: array attribute");

  if (tblgen_padding && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_padding))))
    return emitError(loc, "'TinyFusion.conv2d_relu' op ""attribute 'padding' failed to satisfy constraint: array attribute");

  if (tblgen_stride && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_stride))))
    return emitError(loc, "'TinyFusion.conv2d_relu' op ""attribute 'stride' failed to satisfy constraint: array attribute");

  if (tblgen_max_fp && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_max_fp))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_max_fp).getType().isF32()))))
    return emitError(loc, "'TinyFusion.conv2d_relu' op ""attribute 'max_fp' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_min_fp && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_min_fp))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_min_fp).getType().isF32()))))
    return emitError(loc, "'TinyFusion.conv2d_relu' op ""attribute 'min_fp' failed to satisfy constraint: 32-bit float attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2dReluOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange Conv2dReluOp::getBiasTensorMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::LogicalResult Conv2dReluOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.dilation;
       auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dilation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.max_fp;
       auto attr = dict.get("max_fp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for max_fp in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `max_fp` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.min_fp;
       auto attr = dict.get("min_fp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for min_fp in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `min_fp` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.padding;
       auto attr = dict.get("padding");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for padding in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `padding` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
       auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `stride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute Conv2dReluOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.max_fp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("max_fp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.min_fp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("min_fp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.padding;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("padding",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code Conv2dReluOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.max_fp.getAsOpaquePointer()), 
    llvm::hash_value(prop.min_fp.getAsOpaquePointer()), 
    llvm::hash_value(prop.padding.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> Conv2dReluOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "max_fp")
      return prop.max_fp;

    if (name == "min_fp")
      return prop.min_fp;

    if (name == "padding")
      return prop.padding;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void Conv2dReluOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "max_fp") {
       prop.max_fp = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.max_fp)>>(value);
       return;
    }

    if (name == "min_fp") {
       prop.min_fp = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.min_fp)>>(value);
       return;
    }

    if (name == "padding") {
       prop.padding = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.padding)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void Conv2dReluOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.max_fp) attrs.append("max_fp", prop.max_fp);

    if (prop.min_fp) attrs.append("min_fp", prop.min_fp);

    if (prop.padding) attrs.append("padding", prop.padding);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult Conv2dReluOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "dilation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMaxFpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps1(attr, "max_fp", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMinFpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps1(attr, "min_fp", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPaddingAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "padding", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "stride", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult Conv2dReluOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.dilation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.max_fp)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.min_fp)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.padding)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.stride)))
    return ::mlir::failure();
  return ::mlir::success();
}

void Conv2dReluOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.dilation);
  writer.writeAttribute(prop.max_fp);
  writer.writeAttribute(prop.min_fp);
  writer.writeAttribute(prop.padding);
  writer.writeAttribute(prop.stride);
}

::mlir::ArrayAttr Conv2dReluOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOp::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::llvm::APFloat Conv2dReluOp::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

::llvm::APFloat Conv2dReluOp::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

void Conv2dReluOp::setMaxFp(::llvm::APFloat attrValue) {
  getProperties().max_fp = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void Conv2dReluOp::setMinFp(::llvm::APFloat attrValue) {
  getProperties().min_fp = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void Conv2dReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::mlir::FloatAttr max_fp, ::mlir::FloatAttr min_fp) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  if (bias_tensor)
    odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().max_fp = max_fp;
  odsState.getOrAddProperties<Properties>().min_fp = min_fp;
  odsState.addTypes(result);
}

void Conv2dReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::mlir::FloatAttr max_fp, ::mlir::FloatAttr min_fp) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  if (bias_tensor)
    odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().max_fp = max_fp;
  odsState.getOrAddProperties<Properties>().min_fp = min_fp;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2dReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::llvm::APFloat max_fp, ::llvm::APFloat min_fp) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  if (bias_tensor)
    odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().max_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp);
  odsState.getOrAddProperties<Properties>().min_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp);
  odsState.addTypes(result);
}

void Conv2dReluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::llvm::APFloat max_fp, ::llvm::APFloat min_fp) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  if (bias_tensor)
    odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().max_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp);
  odsState.getOrAddProperties<Properties>().min_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2dReluOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv2dReluOp::verifyInvariantsImpl() {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_max_fp = getProperties().max_fp; (void)tblgen_max_fp;
  if (!tblgen_max_fp) return emitOpError("requires attribute 'max_fp'");
  auto tblgen_min_fp = getProperties().min_fp; (void)tblgen_min_fp;
  if (!tblgen_min_fp) return emitOpError("requires attribute 'min_fp'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitOpError("requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_padding, "padding")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps1(*this, tblgen_max_fp, "max_fp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps1(*this, tblgen_min_fp, "min_fp")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2dReluOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void Conv2dReluOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace TinyFusion
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::Conv2dReluOp)

namespace mlir {
namespace TinyFusion {

//===----------------------------------------------------------------------===//
// ::mlir::TinyFusion::Conv2dSiluOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
Conv2dSiluOpGenericAdaptorBase::Conv2dSiluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("TinyFusion.conv2d_silu", odsAttrs.getContext());
}

Conv2dSiluOpGenericAdaptorBase::Conv2dSiluOpGenericAdaptorBase(Conv2dSiluOp op) : Conv2dSiluOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::ArrayAttr Conv2dSiluOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dSiluOpGenericAdaptorBase::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dSiluOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

} // namespace detail
Conv2dSiluOpAdaptor::Conv2dSiluOpAdaptor(Conv2dSiluOp op) : Conv2dSiluOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult Conv2dSiluOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'TinyFusion.conv2d_silu' op ""requires attribute 'dilation'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitError(loc, "'TinyFusion.conv2d_silu' op ""requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'TinyFusion.conv2d_silu' op ""requires attribute 'stride'");

  if (tblgen_dilation && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_dilation))))
    return emitError(loc, "'TinyFusion.conv2d_silu' op ""attribute 'dilation' failed to satisfy constraint: array attribute");

  if (tblgen_padding && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_padding))))
    return emitError(loc, "'TinyFusion.conv2d_silu' op ""attribute 'padding' failed to satisfy constraint: array attribute");

  if (tblgen_stride && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_stride))))
    return emitError(loc, "'TinyFusion.conv2d_silu' op ""attribute 'stride' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

::mlir::LogicalResult Conv2dSiluOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.dilation;
       auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dilation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.padding;
       auto attr = dict.get("padding");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for padding in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `padding` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
       auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `stride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute Conv2dSiluOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.padding;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("padding",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code Conv2dSiluOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.padding.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> Conv2dSiluOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "padding")
      return prop.padding;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void Conv2dSiluOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "padding") {
       prop.padding = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.padding)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void Conv2dSiluOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.padding) attrs.append("padding", prop.padding);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult Conv2dSiluOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "dilation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPaddingAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "padding", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(attr, "stride", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult Conv2dSiluOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.dilation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.padding)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.stride)))
    return ::mlir::failure();
  return ::mlir::success();
}

void Conv2dSiluOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.dilation);
  writer.writeAttribute(prop.padding);
  writer.writeAttribute(prop.stride);
}

::mlir::ArrayAttr Conv2dSiluOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dSiluOp::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dSiluOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

void Conv2dSiluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, ::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.addTypes(result);
}

void Conv2dSiluOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, ::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride) {
  odsState.addOperands(input_tensor);
  odsState.addOperands(weight_tensor);
  odsState.addOperands(bias_tensor);
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  odsState.getOrAddProperties<Properties>().padding = padding;
  odsState.getOrAddProperties<Properties>().stride = stride;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2dSiluOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv2dSiluOp::verifyInvariantsImpl() {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  if (!tblgen_padding) return emitOpError("requires attribute 'padding'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_padding, "padding")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TinyFusionOps0(*this, tblgen_stride, "stride")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TinyFusionOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2dSiluOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void Conv2dSiluOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace TinyFusion
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::Conv2dSiluOp)


#endif  // GET_OP_CLASSES

