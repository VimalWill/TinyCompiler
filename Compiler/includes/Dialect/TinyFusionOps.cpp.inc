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

::mlir::TinyFusion::Conv2dReluOp
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

::mlir::DictionaryAttr Conv2dReluOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getDilationAttr() {
  auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().dilation);
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getPaddingAttr() {
  auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().padding);
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getStrideAttr() {
  auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().stride);
  return attr;
}

::mlir::ArrayAttr Conv2dReluOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::FloatAttr Conv2dReluOpGenericAdaptorBase::getMaxFpAttr() {
  auto attr = ::llvm::cast<::mlir::FloatAttr>(getProperties().max_fp);
  return attr;
}

::llvm::APFloat Conv2dReluOpGenericAdaptorBase::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr Conv2dReluOpGenericAdaptorBase::getMinFpAttr() {
  auto attr = ::llvm::cast<::mlir::FloatAttr>(getProperties().min_fp);
  return attr;
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

::mlir::Operation::operand_range Conv2dReluOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> Conv2dReluOp::getInputTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> Conv2dReluOp::getWeightTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::TensorType> Conv2dReluOp::getBiasTensor() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::TensorType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*operands.begin());
}

::mlir::OpOperand &Conv2dReluOp::getInputTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return getOperation()->getOpOperand(range.first);
}

::mlir::OpOperand &Conv2dReluOp::getWeightTensorMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return getOperation()->getOpOperand(range.first);
}

::mlir::MutableOperandRange Conv2dReluOp::getBiasTensorMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2dReluOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Conv2dReluOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> Conv2dReluOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

::mlir::ArrayAttr Conv2dReluOp::getDilationAttr() {
  return ::llvm::cast<::mlir::ArrayAttr>(getProperties().dilation);
}

::mlir::ArrayAttr Conv2dReluOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOp::getPaddingAttr() {
  return ::llvm::cast<::mlir::ArrayAttr>(getProperties().padding);
}

::mlir::ArrayAttr Conv2dReluOp::getPadding() {
  auto attr = getPaddingAttr();
  return attr;
}

::mlir::ArrayAttr Conv2dReluOp::getStrideAttr() {
  return ::llvm::cast<::mlir::ArrayAttr>(getProperties().stride);
}

::mlir::ArrayAttr Conv2dReluOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::FloatAttr Conv2dReluOp::getMaxFpAttr() {
  return ::llvm::cast<::mlir::FloatAttr>(getProperties().max_fp);
}

::llvm::APFloat Conv2dReluOp::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr Conv2dReluOp::getMinFpAttr() {
  return ::llvm::cast<::mlir::FloatAttr>(getProperties().min_fp);
}

::llvm::APFloat Conv2dReluOp::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

void Conv2dReluOp::setDilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void Conv2dReluOp::setPaddingAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPaddingAttrName(), attr);
}

void Conv2dReluOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void Conv2dReluOp::setMaxFpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(getMaxFpAttrName(), attr);
}

void Conv2dReluOp::setMaxFp(::llvm::APFloat attrValue) {
  (*this)->setAttr(getMaxFpAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue));
}

void Conv2dReluOp::setMinFpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(getMinFpAttrName(), attr);
}

void Conv2dReluOp::setMinFp(::llvm::APFloat attrValue) {
  (*this)->setAttr(getMinFpAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue));
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


#endif  // GET_OP_CLASSES

