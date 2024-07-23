/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TinyFusionOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace TinyFusion {
class Conv2dReluOp;
} // namespace TinyFusion
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TinyFusion {

//===----------------------------------------------------------------------===//
// ::mlir::TinyFusion::Conv2dReluOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class Conv2dReluOpGenericAdaptorBase {
public:
  struct Properties {
    using dilationTy = ::mlir::ArrayAttr;
    dilationTy dilation;

    auto getDilation() {
      auto &propStorage = this->dilation;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setDilation(const ::mlir::ArrayAttr &propValue) {
      this->dilation = propValue;
    }
    using max_fpTy = ::mlir::FloatAttr;
    max_fpTy max_fp;

    auto getMaxFp() {
      auto &propStorage = this->max_fp;
      return ::llvm::cast<::mlir::FloatAttr>(propStorage);
    }
    void setMaxFp(const ::mlir::FloatAttr &propValue) {
      this->max_fp = propValue;
    }
    using min_fpTy = ::mlir::FloatAttr;
    min_fpTy min_fp;

    auto getMinFp() {
      auto &propStorage = this->min_fp;
      return ::llvm::cast<::mlir::FloatAttr>(propStorage);
    }
    void setMinFp(const ::mlir::FloatAttr &propValue) {
      this->min_fp = propValue;
    }
    using paddingTy = ::mlir::ArrayAttr;
    paddingTy padding;

    auto getPadding() {
      auto &propStorage = this->padding;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setPadding(const ::mlir::ArrayAttr &propValue) {
      this->padding = propValue;
    }
    using strideTy = ::mlir::ArrayAttr;
    strideTy stride;

    auto getStride() {
      auto &propStorage = this->stride;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setStride(const ::mlir::ArrayAttr &propValue) {
      this->stride = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.dilation == this->dilation &&
        rhs.max_fp == this->max_fp &&
        rhs.min_fp == this->min_fp &&
        rhs.padding == this->padding &&
        rhs.stride == this->stride &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Conv2dReluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {});

  Conv2dReluOpGenericAdaptorBase(Conv2dReluOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getDilationAttr();
  ::mlir::ArrayAttr getDilation();
  ::mlir::ArrayAttr getPaddingAttr();
  ::mlir::ArrayAttr getPadding();
  ::mlir::ArrayAttr getStrideAttr();
  ::mlir::ArrayAttr getStride();
  ::mlir::FloatAttr getMaxFpAttr();
  ::llvm::APFloat getMaxFp();
  ::mlir::FloatAttr getMinFpAttr();
  ::llvm::APFloat getMinFp();
};
} // namespace detail
template <typename RangeT>
class Conv2dReluOpGenericAdaptor : public detail::Conv2dReluOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Conv2dReluOpGenericAdaptorBase;
public:
  Conv2dReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Conv2dReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Conv2dReluOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  template <typename LateInst = Conv2dReluOp, typename = std::enable_if_t<std::is_same_v<LateInst, Conv2dReluOp>>>
  Conv2dReluOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInputTensor() {
    return (*getODSOperands(0).begin());
  }

  ValueT getWeightTensor() {
    return (*getODSOperands(1).begin());
  }

  ValueT getBiasTensor() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Conv2dReluOpAdaptor : public Conv2dReluOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Conv2dReluOpGenericAdaptor::Conv2dReluOpGenericAdaptor;
  Conv2dReluOpAdaptor(Conv2dReluOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class Conv2dReluOp : public ::mlir::Op<Conv2dReluOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2dReluOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Conv2dReluOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilation"), ::llvm::StringRef("max_fp"), ::llvm::StringRef("min_fp"), ::llvm::StringRef("padding"), ::llvm::StringRef("stride")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDilationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMaxFpAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMaxFpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMinFpAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMinFpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStrideAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("TinyFusion.conv2d_relu");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInputTensor();
  ::mlir::TypedValue<::mlir::TensorType> getWeightTensor();
  ::mlir::TypedValue<::mlir::TensorType> getBiasTensor();
  ::mlir::OpOperand &getInputTensorMutable();
  ::mlir::OpOperand &getWeightTensorMutable();
  ::mlir::MutableOperandRange getBiasTensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getResult();
  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::ArrayAttr getDilationAttr();
  ::mlir::ArrayAttr getDilation();
  ::mlir::ArrayAttr getPaddingAttr();
  ::mlir::ArrayAttr getPadding();
  ::mlir::ArrayAttr getStrideAttr();
  ::mlir::ArrayAttr getStride();
  ::mlir::FloatAttr getMaxFpAttr();
  ::llvm::APFloat getMaxFp();
  ::mlir::FloatAttr getMinFpAttr();
  ::llvm::APFloat getMinFp();
  void setDilationAttr(::mlir::ArrayAttr attr);
  void setPaddingAttr(::mlir::ArrayAttr attr);
  void setStrideAttr(::mlir::ArrayAttr attr);
  void setMaxFpAttr(::mlir::FloatAttr attr);
  void setMaxFp(::llvm::APFloat attrValue);
  void setMinFpAttr(::mlir::FloatAttr attr);
  void setMinFp(::llvm::APFloat attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input_tensor, Value weight_tensor, Value bias_tensor, ArrayAttr dilation, ArrayAttr padding, ArrayAttr stride, FloatAttr max_fp, FloatAttr min_fp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::mlir::FloatAttr max_fp, ::mlir::FloatAttr min_fp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::mlir::FloatAttr max_fp, ::mlir::FloatAttr min_fp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::llvm::APFloat max_fp, ::llvm::APFloat min_fp);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_tensor, ::mlir::Value weight_tensor, /*optional*/::mlir::Value bias_tensor, ::mlir::ArrayAttr dilation, ::mlir::ArrayAttr padding, ::mlir::ArrayAttr stride, ::llvm::APFloat max_fp, ::llvm::APFloat min_fp);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace TinyFusion
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TinyFusion::Conv2dReluOp)


#endif  // GET_OP_CLASSES

