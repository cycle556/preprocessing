from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import difflib
from data_extractor import ExtractedField


@dataclass
class ComplianceCheckResult:
    field_name: str
    original_consistency: bool
    source_traceable: bool
    overall_pass: bool
    issues: List[str]
    suggestions: List[str]


class InsuranceComplianceChecker:
    def __init__(self):
        self.min_similarity_threshold = 0.9
    
    def check_original_consistency(self, extracted_value: str, 
                                  source_text: str) -> Tuple[bool, float, str]:
        if not extracted_value or not source_text:
            return False, 0.0, "提取值或原文为空"
        
        similarity = difflib.SequenceMatcher(None, extracted_value, source_text).ratio()
        
        if extracted_value in source_text:
            return True, 1.0, "提取值完全匹配原文"
        
        if similarity >= self.min_similarity_threshold:
            return True, similarity, f"提取值与原文高度相似 (相似度: {similarity:.2f})"
        
        return False, similarity, f"提取值与原文相似度不足 (相似度: {similarity:.2f})"
    
    def check_source_traceable(self, source_metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required_fields = ["source"]
        missing_fields = []
        
        for field in required_fields:
            if field not in source_metadata or not source_metadata[field]:
                missing_fields.append(field)
        
        if missing_fields:
            return False, missing_fields
        
        return True, []
    
    def check_field(self, field: ExtractedField) -> ComplianceCheckResult:
        issues = []
        suggestions = []
        
        original_consistency, similarity, consistency_message = self.check_original_consistency(
            field.value, field.source_text
        )
        
        if not original_consistency:
            issues.append(consistency_message)
            suggestions.append("请重新检查提取逻辑，确保100%从原文提取")
        
        source_traceable, missing_fields = self.check_source_traceable(field.source_metadata)
        
        if not source_traceable:
            issues.append(f"来源信息不完整，缺少字段: {', '.join(missing_fields)}")
            suggestions.append("请补充文档来源信息")
        
        overall_pass = original_consistency and source_traceable
        
        return ComplianceCheckResult(
            field_name=field.field_name,
            original_consistency=original_consistency,
            source_traceable=source_traceable,
            overall_pass=overall_pass,
            issues=issues,
            suggestions=suggestions
        )
    
    def check_all_fields(self, fields: List[ExtractedField]) -> List[ComplianceCheckResult]:
        results = []
        for field in fields:
            results.append(self.check_field(field))
        return results
    
    def format_source_info(self, source_metadata: Dict[str, Any]) -> str:
        parts = []
        
        if "source" in source_metadata:
            parts.append(f"文档: {source_metadata['source']}")
        if "chapter" in source_metadata and source_metadata["chapter"]:
            parts.append(f"章节: {source_metadata['chapter']}")
        if "section" in source_metadata and source_metadata["section"]:
            parts.append(f"条款: {source_metadata['section']}")
        if "page_number" in source_metadata:
            parts.append(f"页码: {source_metadata['page_number']}")
        
        return " | ".join(parts) if parts else "来源信息未知"
    
    def generate_compliance_report(self, check_results: List[ComplianceCheckResult]) -> Dict[str, Any]:
        total_fields = len(check_results)
        passed_fields = sum(1 for r in check_results if r.overall_pass)
        failed_fields = total_fields - passed_fields
        
        all_issues = []
        all_suggestions = []
        
        for result in check_results:
            all_issues.extend([f"[{result.field_name}] {issue}" for issue in result.issues])
            all_suggestions.extend([f"[{result.field_name}] {suggestion}" for suggestion in result.suggestions])
        
        return {
            "summary": {
                "total_fields": total_fields,
                "passed_fields": passed_fields,
                "failed_fields": failed_fields,
                "pass_rate": passed_fields / total_fields if total_fields > 0 else 0
            },
            "details": check_results,
            "all_issues": all_issues,
            "all_suggestions": all_suggestions,
            "overall_compliant": failed_fields == 0
        }
