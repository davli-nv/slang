// slang-ir-hlsl-legalize.cpp
#include "slang-ir-hlsl-legalize.h"

#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

#include <functional>

namespace Slang
{

void searchChildrenForForceVarIntoStructTemporarily(IRModule* module, IRInst* inst)
{
    for (auto child : inst->getChildren())
    {
        switch (child->getOp())
        {
        case kIROp_Block:
            {
                searchChildrenForForceVarIntoStructTemporarily(module, child);
                break;
            }
        case kIROp_Call:
            {
                auto call = as<IRCall>(child);
                for (UInt i = 0; i < call->getArgCount(); i++)
                {
                    auto arg = call->getArg(i);
                    const bool isForcedStruct = arg->getOp() == kIROp_ForceVarIntoStructTemporarily;
                    const bool isForcedRayPayloadStruct =
                        arg->getOp() == kIROp_ForceVarIntoRayPayloadStructTemporarily;
                    if (!(isForcedStruct || isForcedRayPayloadStruct))
                        continue;
                    auto forceStructArg = arg->getOperand(0);
                    auto forceStructBaseType =
                        (IRType*)(forceStructArg->getDataType()->getOperand(0));
                    IRBuilder builder(call);
                    if (forceStructBaseType->getOp() == kIROp_StructType)
                    {
                        call->setArg(i, arg->getOperand(0));
                        if (isForcedRayPayloadStruct)
                            builder.addRayPayloadDecoration(forceStructBaseType);
                        continue;
                    }

                    // When `__forceVarIntoStructTemporarily` is called with a non-struct type
                    // parameter, we create a temporary struct and copy the parameter into the
                    // struct. This struct is then subsituted for the return of
                    // `__forceVarIntoStructTemporarily`. Optionally, if
                    // `__forceVarIntoStructTemporarily` is a parameter to a side effect type
                    // (`ref`, `out`, `inout`) we copy the struct back into our original non-struct
                    // parameter.

                    const auto typeNameHint = isForcedRayPayloadStruct
                                                  ? "RayPayload_t"
                                                  : "ForceVarIntoStructTemporarily_t";
                    const auto varNameHint =
                        isForcedRayPayloadStruct ? "rayPayload" : "forceVarIntoStructTemporarily";

                    builder.setInsertBefore(call->getCallee());
                    auto structType = builder.createStructType();
                    StringBuilder structName;
                    builder.addNameHintDecoration(structType, UnownedStringSlice(typeNameHint));
                    if (isForcedRayPayloadStruct)
                        builder.addRayPayloadDecoration(structType);

                    auto elementBufferKey = builder.createStructKey();
                    builder.addNameHintDecoration(elementBufferKey, UnownedStringSlice("data"));
                    auto _dataField = builder.createStructField(
                        structType,
                        elementBufferKey,
                        forceStructBaseType);

                    builder.setInsertBefore(call);
                    auto structVar = builder.emitVar(structType);
                    builder.addNameHintDecoration(structVar, UnownedStringSlice(varNameHint));
                    builder.emitStore(
                        builder.emitFieldAddress(
                            builder.getPtrType(_dataField->getFieldType()),
                            structVar,
                            _dataField->getKey()),
                        builder.emitLoad(forceStructArg));

                    arg->replaceUsesWith(structVar);
                    arg->removeAndDeallocate();

                    auto argType = call->getCallee()->getDataType()->getOperand(i + 1);
                    if (!isPtrLikeOrHandleType(argType))
                        continue;

                    builder.setInsertAfter(call);
                    builder.emitStore(
                        forceStructArg,
                        builder.emitFieldAddress(
                            builder.getPtrType(_dataField->getFieldType()),
                            structVar,
                            _dataField->getKey()));
                }
                break;
            }
        }
    }
}

void legalizeNonStructParameterToStructForHLSL(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->getOp() != kIROp_Func)
            continue;
        searchChildrenForForceVarIntoStructTemporarily(module, globalInst);
    }
}

static bool isSamplePositionSemantic(const UnownedStringSlice& s)
{
    return s.caseInsensitiveEquals(UnownedStringSlice("SV_SamplePosition"));
}

static bool paramHasSamplePositionByDecor(IRInst* inst)
{
    if (auto sem = inst->findDecoration<IRSemanticDecoration>())
        return isSamplePositionSemantic(sem->getSemanticName());
    return false;
}

static bool layoutHasSamplePosition(IRVarLayout* vlay)
{
    if (!vlay)
        return false;
    if (auto sys = vlay->findSystemValueSemanticAttr())
        return isSamplePositionSemantic(UnownedStringSlice(sys->getName()));
    return false;
}

static bool fieldHasSamplePosition(IRStructField* field, IRVarLayout* fieldLayout)
{
    // By key decoration
    if (auto sem = field->getKey()->findDecoration<IRSemanticDecoration>())
        if (isSamplePositionSemantic(sem->getSemanticName()))
            return true;
    // By field var-layout
    return layoutHasSamplePosition(fieldLayout);
}

static bool typeOrLayoutHasSamplePosition(IRStructType* st, IRStructTypeLayout* stLayout)
{
    Index i = 0;
    for (auto field : st->getFields())
    {
        IRVarLayout* fldLayout = stLayout ? stLayout->getFieldLayout(i) : nullptr;
        if (fieldHasSamplePosition(field, fldLayout))
            return true;

        // Recurse into nested structs
        if (auto innerSt = as<IRStructType>(field->getFieldType()))
        {
            auto innerStLayout =
                stLayout ? as<IRStructTypeLayout>(fldLayout ? fldLayout->getTypeLayout() : nullptr)
                         : nullptr;
            if (typeOrLayoutHasSamplePosition(innerSt, innerStLayout))
                return true;
        }
        ++i;
    }
    return false;
}

static bool paramHasSamplePosition(IRParam* p)
{
    // 1) Direct semantic on param
    if (paramHasSamplePositionByDecor(p))
        return true;

    // 2) System-value semantic on param var layout
    IRVarLayout* vlay = nullptr;
    if (auto layDec = p->findDecoration<IRLayoutDecoration>())
        vlay = as<IRVarLayout>(layDec->getLayout());
    if (layoutHasSamplePosition(vlay))
        return true;

    IRType* paramType = p->getDataType();
    IRType* valueType = paramType;
    IRPtrTypeBase* paramPtrType = as<IRPtrTypeBase>(paramType);
    if (paramPtrType) // parameter is passed by const reference
        valueType = paramPtrType->getValueType();
    // 3) If struct, search fields (including nested)
    if (auto st = as<IRStructType>(valueType))
    {
        auto stLayout = vlay ? as<IRStructTypeLayout>(vlay->getTypeLayout()) : nullptr;
        if (typeOrLayoutHasSamplePosition(st, stLayout))
            return true;
    }
    return false;
}

// Scan functions
static void findParamsWithSamplePosition(IRModule* m, List<std::pair<IRFunc*, IRParam*>>& out)
{
    for (auto g : m->getGlobalInsts())
    {
        auto f = as<IRFunc>(g);
        if (!f)
            continue;
        auto first = f->getFirstBlock();
        if (!first)
            continue;

        for (auto p = first->getFirstParam(); p; p = p->getNextParam())
        {
            if (paramHasSamplePosition(p))
                out.add({f, p});
        }
    }
}

// Legalize SV_SamplePosition to GetRenderTargetSamplePosition(SV_SampleIndex) for D3D/HLSL targets
void legalizeSamplePosition(IRModule* module)
{
    List<std::pair<IRFunc*, IRParam*>> list;
    findParamsWithSamplePosition(module, list);
    for (auto globalInst : module->getGlobalInsts())
    {
        auto func = as<IRGlobalValueWithCode>(globalInst);
        if (!func)
            continue;
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getChildren())
            {
                IRInst* load = as<IRLoad>(inst);
                if (!load)
                    continue;
                for (UInt i = 0; i < load->getOperandCount(); i++)
                {
                    auto operand = load->getOperand(i);
                    IRSemanticDecoration* semanticDecor =
                        operand->findDecoration<IRSemanticDecoration>();
                    if (!semanticDecor ||
                        !semanticDecor->getSemanticName().caseInsensitiveEquals(
                            UnownedStringSlice ("sv_sampleposition")))
                        continue;

                    // find GetRenderTargetSamplePosition in coreModules
                    IRInst* samplePositionFunc = nullptr;
                    for (auto coreModule : module->getSession()->coreModules)
                    {
                        for (auto globalInst : coreModule->getIRModule()->getGlobalInsts())
                        {
                            auto func = as<IRGlobalValueWithCode>(globalInst);
                            if (!func)
                                continue;
                            IRNameHintDecoration* nameHintDecor =
                                func->findDecoration<IRNameHintDecoration>();
                            if (nameHintDecor && 
                                nameHintDecor->getName() == UnownedStringSlice("GetRenderTargetSamplePosition"))
                            {
                                samplePositionFunc = func;
                                break;
                            }    
                        }
                        if (samplePositionFunc)
                            break;
                    }

                    IRBuilder builder(load);
                    builder.setInsertBefore(load);              
                    
                    IRType* type = builder.getVectorType(builder.getFloatType(), 2);
                    IRInst* sampleIndexVar;
                    //IRInst* sampleIndex = builder.emitLoad(sampleIndexVar);
                    IRInst* sampleIndex = builder.getIntValue(builder.getUIntType(), 3);
                    IRCall* call = builder.emitCallInst(type, samplePositionFunc, 1, &sampleIndex);

                    load->replaceUsesWith(call);

                    //semanticDecor->removeAndDeallocate();
                    // 
                    //builder.setInsertBefore(semanticDecor);
                    //semanticDecor->setOperand(0, builder.getStringValue(UnownedStringSlice("IGNORE_SEMANTIC")));
                }
            }
        }

        bool needFuncTypeFixup = false;
        for (auto pp = func->getFirstBlock()->getFirstParam(); pp;)
        {
            auto next = pp->getNextParam();
            IRSemanticDecoration* semanticDecor = pp->findDecoration<IRSemanticDecoration>();
            if (!semanticDecor || 
                !semanticDecor->getSemanticName().caseInsensitiveEquals(UnownedStringSlice("sv_sampleposition")))
            {
                pp = next;
                continue;
            }
            pp->removeAndDeallocate();
            needFuncTypeFixup = true;
            pp = next;
        }
        if (needFuncTypeFixup)
            fixUpFuncType(as<IRFunc>(func));
    }

    
}

} // namespace Slang
